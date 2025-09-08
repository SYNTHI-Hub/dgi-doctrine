import fitz  # PyMuPDF
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from django.core.files.base import ContentFile
from django.utils import timezone
from django.db import transaction
from datetime import timedelta
from decimal import Decimal
from doctrine.models import (
    Document, DocumentContent, Topic, Section,
    Paragraph, Table, Theme
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Classe de données pour les résultats d'extraction"""
    success: bool
    content: str = ""
    topics: List[Dict] = None
    sections: List[Dict] = None
    paragraphs: List[Dict] = None
    tables: List[Dict] = None
    metadata: Dict = None
    errors: List[str] = None
    processing_time: float = 0.0

    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.sections is None:
            self.sections = []
        if self.paragraphs is None:
            self.paragraphs = []
        if self.tables is None:
            self.tables = []
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []


class ContentExtractorInterface(ABC):
    """Interface pour les extracteurs de contenu"""

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        pass

    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        pass


class PDFContentExtractor(ContentExtractorInterface):
    """Extracteur de contenu pour les fichiers PDF"""

    def __init__(self):
        self.supported_formats = {'.pdf'}

    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in self.supported_formats

    def extract(self, file_path: str) -> ExtractionResult:
        """Extraction du contenu PDF avec PyMuPDF"""
        start_time = timezone.now()
        result = ExtractionResult(success=False)

        try:
            doc = fitz.open(file_path)

            full_text = ""
            page_contents = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                page_contents.append({
                    'page_number': page_num + 1,
                    'content': page_text,
                    'bbox': [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1]
                })
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_contents': page_contents
            }

            # Extraction des structures (titres, sections, etc.)
            topics = self._extract_topics(doc, page_contents)
            sections = self._extract_sections(page_contents)
            paragraphs = self._extract_paragraphs(page_contents)
            tables = self._extract_tables(doc)

            result.success = True
            result.content = full_text
            result.topics = topics
            result.sections = sections
            result.paragraphs = paragraphs
            result.tables = tables
            result.metadata = metadata

            doc.close()

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction PDF: {str(e)}")
            result.errors.append(f"Erreur d'extraction PDF: {str(e)}")

        finally:
            end_time = timezone.now()
            result.processing_time = (end_time - start_time).total_seconds()

        return result

    def _extract_topics(self, doc, page_contents: List[Dict]) -> List[Dict]:
        """Extraction des topics basée sur la structure des titres"""
        topics = []
        current_topic = None
        topic_patterns = [
            r'^CHAPITRE\s+[IVXLCDM]+|^CHAPTER\s+\d+',
            r'^TITRE\s+[IVXLCDM]+|^TITLE\s+\d+',
            r'^PARTIE\s+[IVXLCDM]+|^PART\s+\d+',
            r'^\d+\.\s+[A-Z][^.]+$',
            r'^[A-Z][A-Z\s]{10,}$'
        ]

        for page_data in page_contents:
            lines = page_data['content'].split('\n')

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Détection des titres de topics
                for pattern in topic_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        if current_topic:
                            topics.append(current_topic)

                        current_topic = {
                            'title': line,
                            'topic_type': self._determine_topic_type(line),
                            'start_page': page_data['page_number'],
                            'content': line + '\n',
                            'order_index': len(topics),
                            'level': self._determine_topic_level(line)
                        }
                        break
                else:

                    if current_topic:
                        current_topic['content'] += line + '\n'

        if current_topic:
            topics.append(current_topic)

        return topics

    def _extract_sections(self, page_contents: List[Dict]) -> List[Dict]:
        """Extraction des sections"""
        sections = []
        section_patterns = [
            r'^\d+\.\d+\s+[A-Z]',
            r'^[A-Z][a-z]+\s*:',
            r'^[IVX]+\.\s+[A-Z]'
        ]

        for page_data in page_contents:
            lines = page_data['content'].split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Détection des titres de sections
                for pattern in section_patterns:
                    if re.match(pattern, line):
                        if current_section:
                            sections.append(current_section)

                        current_section = {
                            'title': line,
                            'section_type': self._determine_section_type(line),
                            'start_page': page_data['page_number'],
                            'content': line + '\n',
                            'order_index': len(sections)
                        }
                        break
                else:
                    if current_section:
                        current_section['content'] += line + '\n'

            if current_section:
                sections.append(current_section)

        return sections

    def _extract_paragraphs(self, page_contents: List[Dict]) -> List[Dict]:
        """Extraction des paragraphes"""
        paragraphs = []

        for page_data in page_contents:
            text = page_data['content']
            # Diviser le texte en paragraphes
            para_texts = re.split(r'\n\s*\n', text)

            for i, para_text in enumerate(para_texts):
                para_text = para_text.strip()
                if len(para_text) > 50:  # Ignorer les paragraphes trop courts
                    paragraphs.append({
                        'content': para_text,
                        'paragraph_type': self._determine_paragraph_type(para_text),
                        'order_index': len(paragraphs),
                        'start_page': page_data['page_number'],
                        'word_count': len(para_text.split()),
                        'sentence_count': len(re.split(r'[.!?]+', para_text))
                    })

        return paragraphs

    def _extract_tables(self, doc) -> List[Dict]:
        """Extraction des tableaux"""
        tables = []

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Recherche de tableaux via les structures
            # Recherche de tableaux via les structures (selon version PyMuPDF)
            try:
                table_candidates_obj = page.find_tables()
            except Exception:
                table_candidates_obj = None

            if not table_candidates_obj:
                continue

            table_iterable = getattr(table_candidates_obj, 'tables', table_candidates_obj)

            for i, table in enumerate(table_iterable):
                try:
                    # Extraction des données du tableau
                    table_data = table.extract()

                    if table_data and len(table_data) > 1:  # Au moins 2 lignes
                        headers = table_data[0] if table_data else []
                        data_rows = table_data[1:] if len(table_data) > 1 else []

                        tables.append({
                            'title': f'Tableau {len(tables) + 1} - Page {page_num + 1}',
                            'table_type': 'data',
                            'headers': headers,
                            'data': {
                                'rows': data_rows,
                                'raw_data': table_data
                            },
                            'row_count': len(data_rows),
                            'column_count': len(headers) if headers else 0,
                            'start_page': page_num + 1,
                            'order_index': len(tables),
                            'bbox_coordinates': {
                                'x0': float(getattr(table.bbox, 'x0', table.bbox[0] if hasattr(table.bbox, '__getitem__') else 0.0)),
                                'y0': float(getattr(table.bbox, 'y0', table.bbox[1] if hasattr(table.bbox, '__getitem__') else 0.0)),
                                'x1': float(getattr(table.bbox, 'x1', table.bbox[2] if hasattr(table.bbox, '__getitem__') else 0.0)),
                                'y1': float(getattr(table.bbox, 'y1', table.bbox[3] if hasattr(table.bbox, '__getitem__') else 0.0))
                            },
                            'extraction_confidence': 0.8,
                            'extraction_method': 'pymupdf_auto'
                        })

                except Exception as e:
                    logger.warning(f"Erreur extraction tableau page {page_num + 1}: {str(e)}")
                    continue

        return tables

    def _determine_topic_type(self, title: str) -> str:
        """Détermine le type de topic basé sur le titre"""
        title_lower = title.lower()
        if 'chapitre' in title_lower or 'chapter' in title_lower:
            return 'chapter'
        elif 'section' in title_lower:
            return 'section'
        elif 'introduction' in title_lower:
            return 'introduction'
        elif 'conclusion' in title_lower:
            return 'conclusion'
        elif 'annexe' in title_lower or 'appendix' in title_lower:
            return 'appendix'
        return 'section'

    def _determine_topic_level(self, title: str) -> int:
        """Détermine le niveau hiérarchique du topic"""
        if re.match(r'^CHAPITRE|^CHAPTER|^PARTIE|^PART', title, re.IGNORECASE):
            return 1
        elif re.match(r'^\d+\.\s+', title):
            return 2
        elif re.match(r'^\d+\.\d+\s+', title):
            return 3
        return 2

    def _determine_section_type(self, title: str) -> str:
        """Détermine le type de section"""
        if ':' in title:
            return 'heading'
        elif re.match(r'^\d+\.\d+', title):
            return 'paragraph'
        return 'paragraph'

    def _determine_paragraph_type(self, content: str) -> str:
        """Détermine le type de paragraphe"""
        content_lower = content.lower()
        if content_lower.startswith('note'):
            return 'note'
        elif content_lower.startswith('attention') or content_lower.startswith('warning'):
            return 'warning'
        elif content_lower.startswith('exemple') or content_lower.startswith('example'):
            return 'example'
        elif '"' in content and content.count('"') >= 2:
            return 'quote'
        return 'normal'


class WordContentExtractor(ContentExtractorInterface):
    """Extracteur de contenu pour les fichiers Word"""

    def __init__(self):
        self.supported_formats = {'.doc', '.docx'}

    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in self.supported_formats

    def extract(self, file_path: str) -> ExtractionResult:
        """Extraction du contenu Word: tentative via PyMuPDF sinon échec gracieux.
        Par défaut, privilégier PDF; DOC/DOCX supportés si MuPDF le permet.
        """
        start_time = timezone.now()
        result = ExtractionResult(success=False)

        try:
            # Utiliser la même logique que PDF si MuPDF peut ouvrir le fichier
            try:
                _doc = fitz.open(file_path)
                _doc.close()
            except Exception:
                raise RuntimeError("Format Word non supporté par PyMuPDF sur cette plateforme.")

            pdf_extractor = PDFContentExtractor()
            result = pdf_extractor.extract(file_path)

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction Word: {str(e)}")
            result.errors.append(f"Erreur d'extraction Word: {str(e)}")

        finally:
            end_time = timezone.now()
            result.processing_time = (end_time - start_time).total_seconds()

        return result


class DocumentProcessorService:
    """Service principal de traitement des documents"""

    def __init__(self):
        self.extractors = [
            PDFContentExtractor(),
            WordContentExtractor()
        ]

    def get_extractor(self, file_extension: str) -> Optional[ContentExtractorInterface]:
        """Récupère l'extracteur approprié pour le type de fichier"""
        for extractor in self.extractors:
            if extractor.supports_format(file_extension):
                return extractor
        return None

    @transaction.atomic
    def process_document(self, document: Document) -> bool:
        """Traite un document complet et sauvegarde les résultats"""
        try:
            # Mettre à jour le statut
            document.status = Document.StatusChoices.PROCESSING
            document.save(update_fields=['status'])

            # Obtenir l'extracteur approprié
            file_extension = document.file_type
            extractor = self.get_extractor(f'.{file_extension}')

            if not extractor:
                raise ValueError(f"Aucun extracteur disponible pour le type: {file_extension}")

            # Extraction du contenu
            file_path = document.file_path.path
            extraction_result = extractor.extract(file_path)

            if not extraction_result.success:
                raise Exception(f"Échec de l'extraction: {', '.join(extraction_result.errors)}")

            # Sauvegarde du contenu principal
            content_data = self._prepare_content_data(extraction_result, document)
            self._save_document_content(document, content_data)

            # Sauvegarde des structures
            self._save_topics(document, extraction_result.topics)
            self._save_sections(document, extraction_result.sections)
            self._save_paragraphs(document, extraction_result.paragraphs)
            self._save_tables(document, extraction_result.tables)

            # Mise à jour du document
            document.status = Document.StatusChoices.PROCESSED
            document.extraction_metadata = extraction_result.metadata
            document.processing_log.append({
                'timestamp': timezone.now().isoformat(),
                'action': 'content_extraction',
                'success': True,
                'processing_time': extraction_result.processing_time,
                'extractor': extractor.__class__.__name__
            })
            document.save()

            logger.info(f"Document {document.id} traité avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur lors du traitement du document {document.id}: {str(e)}")

            # Mettre le document en erreur
            document.status = Document.StatusChoices.ERROR
            document.processing_log.append({
                'timestamp': timezone.now().isoformat(),
                'action': 'content_extraction',
                'success': False,
                'error': str(e)
            })
            document.save()

            return False

    def _prepare_content_data(self, extraction_result: ExtractionResult, document: Document) -> Dict:
        """Prépare les données de contenu"""
        clean_content = self._clean_text(extraction_result.content)

        return {
            'raw_content': extraction_result.content,
            'clean_content': clean_content,
            'structured_content': {
                'topics_count': len(extraction_result.topics),
                'sections_count': len(extraction_result.sections),
                'paragraphs_count': len(extraction_result.paragraphs),
                'tables_count': len(extraction_result.tables)
            },
            'content_type': DocumentContent.ContentType.STRUCTURED,
            'extraction_method': DocumentContent.ExtractionMethod.AUTOMATIC,
            'processing_status': DocumentContent.ProcessingStatus.COMPLETED,
            'extraction_confidence': Decimal('0.8500'),
            'word_count': len(clean_content.split()),
            'character_count': len(clean_content),
            'sentence_count': len(re.split(r'[.!?]+', clean_content)),
            'paragraph_count': len(extraction_result.paragraphs),
            'page_count': extraction_result.metadata.get('page_count', 0),
            'table_count': len(extraction_result.tables),
            'processed_at': timezone.now(),
            'processing_duration': timedelta(seconds=extraction_result.processing_time)
        }

    def _save_document_content(self, document: Document, content_data: Dict):
        """Sauvegarde le contenu principal du document"""
        content, created = DocumentContent.objects.get_or_create(
            document=document,
            defaults=content_data
        )

        if not created:
            for key, value in content_data.items():
                setattr(content, key, value)
            content.save()

    def _save_topics(self, document: Document, topics_data: List[Dict]):
        """Sauvegarde les topics"""
        # Supprimer les anciens topics
        document.topics.all().delete()

        for topic_data in topics_data:
            Topic.objects.create(
                document=document,
                title=topic_data.get('title', ''),
                content=topic_data.get('content', ''),
                topic_type=topic_data.get('topic_type', Topic.TopicType.SECTION),
                order_index=topic_data.get('order_index', 0),
                level=topic_data.get('level', 1),
                start_page=topic_data.get('start_page'),
                word_count=len(topic_data.get('content', '').split())
            )

    def _save_sections(self, document: Document, sections_data: List[Dict]):
        """Sauvegarde les sections"""
        topics = list(document.topics.all().order_by('order_index'))
        # Créer un topic par défaut si aucun n'existe mais que des sections sont détectées
        if not topics and sections_data:
            default_topic = Topic.objects.create(
                document=document,
                title='Contenu',
                content='Sections détectées',
                topic_type=Topic.TopicType.SECTION,
                order_index=0,
                level=1,
                start_page=sections_data[0].get('start_page') if sections_data else None
            )
            topics = [default_topic]

        def find_topic_for_page(start_page: Optional[int]) -> Optional[Topic]:
            if not topics:
                return None
            if not start_page:
                return topics[0]
            candidates = [t for t in topics if t.start_page and t.start_page <= start_page]
            return candidates[-1] if candidates else topics[0]

        for section_data in sections_data:
            topic = find_topic_for_page(section_data.get('start_page'))
            if topic:
                Section.objects.create(
                    topic=topic,
                    title=section_data.get('title', ''),
                    content=section_data.get('content', ''),
                    section_type=section_data.get('section_type', Section.SectionType.PARAGRAPH),
                    order_index=section_data.get('order_index', 0),
                    start_page=section_data.get('start_page'),
                    word_count=len(section_data.get('content', '').split())
                )

    def _save_paragraphs(self, document: Document, paragraphs_data: List[Dict]):
        """Sauvegarde les paragraphes"""
        sections = list(Section.objects.filter(topic__document=document).order_by('start_page', 'order_index'))
        # Créer une section par défaut si aucune n'existe mais que des paragraphes sont détectés
        if not sections and paragraphs_data:
            topic = Topic.objects.create(
                document=document,
                title='Contenu',
                content='Paragraphes détectés',
                topic_type=Topic.TopicType.SECTION,
                order_index=0,
                level=1,
                start_page=paragraphs_data[0].get('start_page') if paragraphs_data else None
            )
            section = Section.objects.create(
                topic=topic,
                title='Paragraphe',
                content='',
                section_type=Section.SectionType.PARAGRAPH,
                order_index=0,
                start_page=paragraphs_data[0].get('start_page') or 1,
                word_count=0
            )
            sections = [section]

        def find_section_for_page(start_page: Optional[int]) -> Optional[Section]:
            if not sections:
                return None
            if not start_page:
                return sections[0]
            # Find the last section with start_page <= paragraph start_page
            candidates = [s for s in sections if s.start_page and s.start_page <= start_page]
            return candidates[-1] if candidates else sections[0]

        for para_data in paragraphs_data:
            section = find_section_for_page(para_data.get('start_page'))
            if section:
                Paragraph.objects.create(
                    section=section,
                    content=para_data.get('content', ''),
                    paragraph_type=para_data.get('paragraph_type', Paragraph.ParagraphType.NORMAL),
                    order_index=para_data.get('order_index', 0),
                    word_count=para_data.get('word_count', 0),
                    sentence_count=para_data.get('sentence_count', 0)
                )

    def _save_tables(self, document: Document, tables_data: List[Dict]):
        """Sauvegarde les tableaux et les associe à une section la plus proche"""
        sections = list(Section.objects.filter(topic__document=document).order_by('start_page', 'order_index'))

        def ensure_tables_section(start_page: Optional[int]) -> Section:
            # Retourne une section correspondant à la page ou crée une section par défaut
            if sections:
                if start_page:
                    candidates = [s for s in sections if s.start_page and s.start_page <= start_page]
                    return candidates[-1] if candidates else sections[0]
                return sections[0]
            # Créer un topic et une section par défaut si aucune section n'existe
            topic = Topic.objects.create(
                document=document,
                title='Tables',
                content='Tables extraites',
                topic_type=Topic.TopicType.SECTION,
                order_index=0,
                level=1,
                start_page=start_page
            )
            section = Section.objects.create(
                topic=topic,
                title='Tables',
                content='',
                section_type=Section.SectionType.PARAGRAPH,
                order_index=0,
                start_page=start_page or 1,
                word_count=0
            )
            sections.append(section)
            return section

        for table_data in tables_data:
            section = ensure_tables_section(table_data.get('start_page'))
            Table.objects.create(
                section=section,
                title=table_data.get('title', ''),
                caption=table_data.get('caption', ''),
                table_type=table_data.get('table_type', Table.TableType.DATA),
                headers=table_data.get('headers', []),
                data=table_data.get('data', {}),
                raw_data=table_data.get('data', {}),
                row_count=table_data.get('row_count', 0),
                column_count=table_data.get('column_count', 0),
                order_index=table_data.get('order_index', 0),
                extraction_confidence=Decimal(str(table_data.get('extraction_confidence', 0.0))),
                extraction_method=table_data.get('extraction_method', ''),
                bbox_coordinates=table_data.get('bbox_coordinates', {}),
                has_header_row=len(table_data.get('headers', [])) > 0
            )

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte extrait tout en préservant la structure (sauts de ligne)."""
        if not text:
            return ''
        # Normaliser les retours chariot
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Supprimer les caractères de contrôle non imprimables
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        # Réduire les séquences d'espaces/tabs mais conserver les \n
        def _normalize_line(line: str) -> str:
            line = re.sub(r'[ \t]+', ' ', line)
            return line.strip()

        lines = [_normalize_line(l) for l in text.split('\n')]
        text = '\n'.join(lines)
        # Limiter les sauts de ligne consécutifs à 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# Singleton du service
document_processor = DocumentProcessorService()


# Fonction utilitaire pour l'usage externe
def process_document_async(document_id: str) -> bool:
    """Fonction utilitaire pour traiter un document de façon asynchrone"""
    try:
        document = Document.objects.get(id=document_id)
        return document_processor.process_document(document)
    except Document.DoesNotExist:
        logger.error(f"Document {document_id} introuvable")
        return False
    except Exception as e:
        logger.error(f"Erreur lors du traitement du document {document_id}: {str(e)}")
        return False