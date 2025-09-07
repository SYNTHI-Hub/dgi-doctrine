from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import FileExtensionValidator, MinValueValidator, MaxValueValidator
from django.utils.translation import gettext_lazy as _
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.utils.text import slugify
from django.urls import reverse
import uuid
import hashlib
import os
from decimal import Decimal
from django.utils import timezone
from django.contrib.auth.models import BaseUserManager


def get_default_time():
    return timezone.now()


class CustomUserManager(BaseUserManager):
    """Manager personnalisé pour le modèle User"""

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('L\'email est obligatoire')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)


class TimeStampedModel(models.Model):
    """Modèle abstrait pour les timestamps"""
    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Date de création"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Date de modification"))

    class Meta:
        abstract = True


class SoftDeleteModel(models.Model):
    """Modèle abstrait pour la suppression logique"""
    is_deleted = models.BooleanField(default=False, verbose_name=_("Supprimé"))
    deleted_at = models.DateTimeField(null=True, blank=True, verbose_name=_("Date de suppression"))
    deleted_by = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, blank=True,
                                   related_name='%(class)s_deleted', verbose_name=_("Supprimé par"))

    class Meta:
        abstract = True

    def soft_delete(self, user=None):
        """Méthode pour effectuer une suppression logique"""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])

    def restore(self):
        """Méthode pour restaurer un élément supprimé"""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
        self.save(update_fields=['is_deleted', 'deleted_at', 'deleted_by'])


class User(AbstractUser, TimeStampedModel):
    """Modèle utilisateur étendu"""

    class RoleChoices(models.TextChoices):
        ADMIN = 'admin', _('Administrateur')
        CONTRIBUTOR = 'contributor', _('Contributeur')
        USER = 'user', _('Utilisateur')
        GUEST = 'guest', _('Invité')
        MODERATOR = 'moderator', _('Modérateur')
        ANALYST = 'analyst', _('Analyste')

    class StatusChoices(models.TextChoices):
        ACTIVE = 'active', _('Actif')
        INACTIVE = 'inactive', _('Inactif')
        SUSPENDED = 'suspended', _('Suspendu')
        PENDING = 'pending', _('En attente')
        BANNED = 'banned', _('Banni')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True, verbose_name=_("Email"))
    first_name = models.CharField(max_length=150, verbose_name=_("Prénom"))
    last_name = models.CharField(max_length=150, verbose_name=_("Nom"))
    role = models.CharField(max_length=20, choices=RoleChoices.choices,
                            default=RoleChoices.USER, verbose_name=_("Rôle"))
    status = models.CharField(max_length=20, choices=StatusChoices.choices,
                              default=StatusChoices.ACTIVE, verbose_name=_("Statut"))

    # Informations personnelles
    phone_number = models.CharField(max_length=20, blank=True, verbose_name=_("Téléphone"))
    birth_date = models.DateField(null=True, blank=True, verbose_name=_("Date de naissance"))
    address = models.TextField(blank=True, verbose_name=_("Adresse"))
    city = models.CharField(max_length=100, blank=True, verbose_name=_("Ville"))
    country = models.CharField(max_length=100, blank=True, verbose_name=_("Pays"))
    postal_code = models.CharField(max_length=20, blank=True, verbose_name=_("Code postal"))

    # Informations professionnelles
    department = models.CharField(max_length=100, blank=True, verbose_name=_("Département"))
    position = models.CharField(max_length=100, blank=True, verbose_name=_("Poste"))
    employee_id = models.CharField(max_length=50, blank=True, unique=True, null=True, verbose_name=_("ID employé"))
    manager = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True,
                                related_name='team_members', verbose_name=_("Manager"))
    hire_date = models.DateField(null=True, blank=True, verbose_name=_("Date d'embauche"))
    salary_grade = models.CharField(max_length=20, blank=True, verbose_name=_("Grade salarial"))

    # Préférences
    profile_picture = models.ImageField(upload_to='profiles/%Y/%m/', blank=True, null=True,
                                        verbose_name=_("Photo de profil"))
    preferred_language = models.CharField(max_length=10, default='fr',
                                          choices=[('fr', 'Français'), ('en', 'English'),
                                                   ('es', 'Español'), ('de', 'Deutsch')],
                                          verbose_name=_("Langue préférée"))
    timezone = models.CharField(max_length=50, default='Europe/Paris', verbose_name=_("Fuseau horaire"))
    theme_preference = models.CharField(max_length=20, default='light',
                                        choices=[('light', 'Clair'), ('dark', 'Sombre'), ('auto', 'Automatique')],
                                        verbose_name=_("Thème préféré"))
    notifications_enabled = models.BooleanField(default=True, verbose_name=_("Notifications activées"))
    email_notifications = models.BooleanField(default=True, verbose_name=_("Notifications par email"))

    # Sécurité et accès
    last_login_ip = models.GenericIPAddressField(null=True, blank=True, verbose_name=_("Dernière IP"))
    failed_login_attempts = models.PositiveIntegerField(default=0, verbose_name=_("Tentatives de connexion échouées"))
    account_locked_until = models.DateTimeField(null=True, blank=True, verbose_name=_("Compte verrouillé jusqu'à"))
    email_verified = models.BooleanField(default=False, verbose_name=_("Email vérifié"))
    email_verification_token = models.CharField(max_length=100, blank=True,
                                                verbose_name=_("Token de vérification email"))
    password_reset_token = models.CharField(max_length=100, blank=True, verbose_name=_("Token de réinitialisation"))
    password_reset_expires = models.DateTimeField(null=True, blank=True, verbose_name=_("Expiration token"))
    two_factor_enabled = models.BooleanField(default=False, verbose_name=_("Authentification à deux facteurs"))
    two_factor_secret = models.CharField(max_length=32, blank=True, verbose_name=_("Secret 2FA"))

    # Accès API et quotas
    api_access_enabled = models.BooleanField(default=False, verbose_name=_("Accès API autorisé"))
    api_key = models.CharField(max_length=100, blank=True, unique=True, null=True, verbose_name=_("Clé API"))
    api_rate_limit = models.PositiveIntegerField(default=1000, verbose_name=_("Limite de taux API"))
    max_documents_upload = models.PositiveIntegerField(default=100, verbose_name=_("Limite upload documents"))
    storage_quota_mb = models.PositiveIntegerField(default=1000, verbose_name=_("Quota stockage (MB)"))
    current_storage_mb = models.PositiveIntegerField(default=0, verbose_name=_("Stockage utilisé (MB)"))

    # Métadonnées
    bio = models.TextField(blank=True, verbose_name=_("Biographie"))
    skills = ArrayField(models.CharField(max_length=100), default=list, blank=True, verbose_name=_("Compétences"))
    interests = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                           verbose_name=_("Centres d'intérêt"))
    social_links = models.JSONField(default=dict, blank=True, verbose_name=_("Liens sociaux"))
    metadata = models.JSONField(default=dict, blank=True, verbose_name=_("Métadonnées"))

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    objects = CustomUserManager()

    class Meta:
        verbose_name = _("Utilisateur")
        verbose_name_plural = _("Utilisateurs")
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['role', 'status']),
            models.Index(fields=['created_at']),
            models.Index(fields=['department', 'position']),
            models.Index(fields=['manager']),
        ]

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

    def get_absolute_url(self):
        return reverse('user-detail', kwargs={'pk': self.pk})

    def is_quota_exceeded(self, file_size_mb):
        """Vérifie si l'ajout d'un fichier dépasserait le quota"""
        return (self.current_storage_mb + file_size_mb) > self.storage_quota_mb

    def can_upload_documents(self):
        """Vérifie si l'utilisateur peut uploader des documents"""
        return self.uploaded_documents.count() < self.max_documents_upload

    def generate_api_key(self):
        """Génère une nouvelle clé API"""
        import secrets
        self.api_key = secrets.token_urlsafe(50)
        self.save(update_fields=['api_key'])
        return self.api_key


class Theme(TimeStampedModel, SoftDeleteModel):
    """Modèle pour les thèmes de classification"""

    class ThemeType(models.TextChoices):
        CATEGORY = 'category', _('Catégorie')
        SUBCATEGORY = 'subcategory', _('Sous-catégorie')
        TAG = 'tag', _('Tag')
        DOMAIN = 'domain', _('Domaine')
        JURISDICTION = 'jurisdiction', _('Juridiction')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, verbose_name=_("Nom"))
    slug = models.SlugField(max_length=220, unique=True, verbose_name=_("Slug"))
    description = models.TextField(blank=True, verbose_name=_("Description"))
    code = models.CharField(max_length=50, unique=True, verbose_name=_("Code"))
    theme_type = models.CharField(max_length=20, choices=ThemeType.choices,
                                  default=ThemeType.CATEGORY, verbose_name=_("Type de thème"))
    parent_theme = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True,
                                     related_name='children', verbose_name=_("Thème parent"))

    # Apparence
    icon = models.CharField(max_length=50, blank=True, verbose_name=_("Icône"))
    color = models.CharField(max_length=7, default='#007bff', verbose_name=_("Couleur"))
    background_color = models.CharField(max_length=7, default='#ffffff', verbose_name=_("Couleur de fond"))
    text_color = models.CharField(max_length=7, default='#000000', verbose_name=_("Couleur du texte"))

    # Organisation et navigation
    order_index = models.PositiveIntegerField(default=0, verbose_name=_("Ordre d'affichage"))
    level = models.PositiveIntegerField(default=1, verbose_name=_("Niveau hiérarchique"))
    path = models.CharField(max_length=500, blank=True, verbose_name=_("Chemin hiérarchique"))

    # État et visibilité
    is_active = models.BooleanField(default=True, verbose_name=_("Actif"))
    is_featured = models.BooleanField(default=False, verbose_name=_("Mis en avant"))
    is_public = models.BooleanField(default=True, verbose_name=_("Public"))

    # Métadonnées et enrichissement
    metadata = models.JSONField(default=dict, blank=True, verbose_name=_("Métadonnées"))
    keywords = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                          verbose_name=_("Mots-clés"))
    aliases = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                         verbose_name=_("Alias"))

    # Statistiques
    documents_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de documents"))
    views_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de vues"))

    # Gestion et supervision
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                                   related_name='created_themes', verbose_name=_("Créé par"))
    managed_by = models.ManyToManyField(User, blank=True, related_name='managed_themes',
                                        verbose_name=_("Géré par"))

    class Meta:
        verbose_name = _("Thème")
        verbose_name_plural = _("Thèmes")
        ordering = ['level', 'order_index', 'name']
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['code']),
            models.Index(fields=['theme_type', 'is_active']),
            models.Index(fields=['parent_theme', 'order_index']),
            models.Index(fields=['is_active', 'is_public']),
        ]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        if not self.code:
            self.code = self.slug.upper().replace('-', '_')

        # Calcul du niveau et du chemin
        if self.parent_theme:
            self.level = self.parent_theme.level + 1
            self.path = f"{self.parent_theme.path}/{self.slug}" if self.parent_theme.path else self.slug
        else:
            self.level = 1
            self.path = self.slug

        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('theme-detail', kwargs={'slug': self.slug})

    def get_children(self):
        """Retourne les thèmes enfants"""
        return self.children.filter(is_active=True, is_deleted=False)

    def get_ancestors(self):
        """Retourne la liste des thèmes parents"""
        ancestors = []
        current = self.parent_theme
        while current:
            ancestors.append(current)
            current = current.parent_theme
        return reversed(ancestors)

    def get_descendants(self):
        """Retourne tous les descendants"""
        descendants = []
        for child in self.get_children():
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants


def default_allowed_file_types():
    return ['pdf', 'doc', 'docx']


class DocumentCategory(TimeStampedModel):
    """Catégories spécifiques aux documents"""

    class CategoryType(models.TextChoices):
        LEGAL = 'legal', _('Juridique')
        ADMINISTRATIVE = 'administrative', _('Administratif')
        TECHNICAL = 'technical', _('Technique')
        REGULATORY = 'regulatory', _('Réglementaire')
        POLICY = 'policy', _('Politique')
        GUIDANCE = 'guidance', _('Guide')
        CASE_STUDY = 'case_study', _('Étude de cas')
        TEMPLATE = 'template', _('Modèle')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True, verbose_name=_("Nom"))
    slug = models.SlugField(max_length=120, unique=True, verbose_name=_("Slug"))
    description = models.TextField(blank=True, verbose_name=_("Description"))
    category_type = models.CharField(max_length=20, choices=CategoryType.choices,
                                     default=CategoryType.LEGAL, verbose_name=_("Type de catégorie"))

    # Apparence
    color = models.CharField(max_length=7, default='#6c757d', verbose_name=_("Couleur"))
    icon = models.CharField(max_length=50, blank=True, verbose_name=_("Icône"))

    # Configuration
    requires_approval = models.BooleanField(default=False, verbose_name=_("Nécessite approbation"))
    default_visibility = models.CharField(max_length=20, default='public',
                                          choices=[('public', 'Public'), ('private', 'Privé'),
                                                   ('restricted', 'Restreint')],
                                          verbose_name=_("Visibilité par défaut"))
    retention_days = models.PositiveIntegerField(null=True, blank=True,
                                                 verbose_name=_("Jours de rétention"))

    # Métadonnées
    metadata_schema = models.JSONField(default=dict, blank=True, verbose_name=_("Schéma de métadonnées"))

    allowed_file_types = ArrayField(
        models.CharField(max_length=10),
        default=default_allowed_file_types,
        verbose_name=_("Types de fichiers autorisés")
    )

    max_file_size_mb = models.PositiveIntegerField(default=50, verbose_name=_("Taille max fichier (MB)"))

    class Meta:
        verbose_name = _("Catégorie de document")
        verbose_name_plural = _("Catégories de documents")
        ordering = ['name']

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class Document(TimeStampedModel, SoftDeleteModel):
    """Modèle principal pour les documents"""

    class StatusChoices(models.TextChoices):
        DRAFT = 'draft', _('Brouillon')
        PENDING = 'pending', _('En attente')
        PROCESSING = 'processing', _('En traitement')
        PROCESSED = 'processed', _('Traité')
        PUBLISHED = 'published', _('Publié')
        ARCHIVED = 'archived', _('Archivé')
        ERROR = 'error', _('Erreur')
        REJECTED = 'rejected', _('Rejeté')

    class VisibilityChoices(models.TextChoices):
        PUBLIC = 'public', _('Public')
        PRIVATE = 'private', _('Privé')
        RESTRICTED = 'restricted', _('Restreint')
        CONFIDENTIAL = 'confidential', _('Confidentiel')
        SECRET = 'secret', _('Secret')

    class LanguageChoices(models.TextChoices):
        FRENCH = 'fr', _('Français')
        ENGLISH = 'en', _('Anglais')
        SPANISH = 'es', _('Espagnol')
        GERMAN = 'de', _('Allemand')
        ITALIAN = 'it', _('Italien')
        PORTUGUESE = 'pt', _('Portugais')

    class PriorityChoices(models.TextChoices):
        LOW = 'low', _('Faible')
        NORMAL = 'normal', _('Normal')
        HIGH = 'high', _('Élevée')
        URGENT = 'urgent', _('Urgente')
        CRITICAL = 'critical', _('Critique')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500, verbose_name=_("Titre"))
    slug = models.SlugField(max_length=520, unique=True, verbose_name=_("Slug"))
    description = models.TextField(blank=True, verbose_name=_("Description"))
    summary = models.TextField(blank=True, verbose_name=_("Résumé"))
    abstract = models.TextField(blank=True, verbose_name=_("Résumé exécutif"))

    # Informations sur le fichier
    original_filename = models.CharField(max_length=255, verbose_name=_("Nom de fichier original"))
    file_path = models.FileField(upload_to='documents/%Y/%m/',
                                 validators=[FileExtensionValidator(allowed_extensions=['pdf', 'doc', 'docx'])],
                                 verbose_name=_("Fichier"))
    file_type = models.CharField(max_length=10, verbose_name=_("Type de fichier"))
    file_size = models.BigIntegerField(verbose_name=_("Taille du fichier (bytes)"))
    file_checksum = models.CharField(max_length=64, verbose_name=_("Empreinte du fichier"))
    mime_type = models.CharField(max_length=100, blank=True, verbose_name=_("Type MIME"))

    # Relations principales
    theme = models.ForeignKey(Theme, on_delete=models.PROTECT, related_name='documents',
                              verbose_name=_("Thème"))
    category = models.ForeignKey(DocumentCategory, on_delete=models.PROTECT,
                                 related_name='documents', verbose_name=_("Catégorie"))
    uploaded_by = models.ForeignKey(User, on_delete=models.PROTECT, related_name='uploaded_documents',
                                    verbose_name=_("Téléchargé par"))

    # Workflow et approbation
    status = models.CharField(max_length=20, choices=StatusChoices.choices,
                              default=StatusChoices.DRAFT, verbose_name=_("Statut"))
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                                    related_name='reviewed_documents', verbose_name=_("Révisé par"))
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                                    related_name='approved_documents', verbose_name=_("Approuvé par"))
    reviewed_at = models.DateTimeField(null=True, blank=True, verbose_name=_("Révisé le"))
    approved_at = models.DateTimeField(null=True, blank=True, verbose_name=_("Approuvé le"))
    rejection_reason = models.TextField(blank=True, verbose_name=_("Raison du rejet"))

    # Visibilité et sécurité
    visibility = models.CharField(max_length=20, choices=VisibilityChoices.choices,
                                  default=VisibilityChoices.PUBLIC, verbose_name=_("Visibilité"))
    security_classification = models.CharField(max_length=20, blank=True, verbose_name=_("Classification sécurité"))
    access_groups = models.ManyToManyField('auth.Group', blank=True, verbose_name=_("Groupes d'accès"))
    authorized_users = models.ManyToManyField(User, blank=True, related_name='authorized_documents',
                                              verbose_name=_("Utilisateurs autorisés"))

    # Métadonnées de contenu
    language = models.CharField(max_length=10, choices=LanguageChoices.choices,
                                default=LanguageChoices.FRENCH, verbose_name=_("Langue"))
    priority = models.CharField(max_length=20, choices=PriorityChoices.choices,
                                default=PriorityChoices.NORMAL, verbose_name=_("Priorité"))

    # Dates importantes
    publication_date = models.DateField(null=True, blank=True, verbose_name=_("Date de publication"))
    effective_date = models.DateField(null=True, blank=True, verbose_name=_("Date d'effet"))
    expiration_date = models.DateField(null=True, blank=True, verbose_name=_("Date d'expiration"))
    review_date = models.DateField(null=True, blank=True, verbose_name=_("Date de révision"))
    last_modified_date = models.DateField(auto_now=True, verbose_name=_("Dernière modification"))

    # Informations légales et réglementaires
    legal_reference = models.CharField(max_length=200, blank=True, verbose_name=_("Référence légale"))
    regulation_number = models.CharField(max_length=100, blank=True, verbose_name=_("Numéro de règlement"))
    jurisdiction = models.CharField(max_length=100, blank=True, verbose_name=_("Juridiction"))
    legal_status = models.CharField(max_length=50, blank=True, verbose_name=_("Statut légal"))
    compliance_requirements = ArrayField(models.CharField(max_length=200), default=list, blank=True,
                                         verbose_name=_("Exigences de conformité"))

    # Compteurs et métriques
    view_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de vues"))
    download_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de téléchargements"))
    share_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de partages"))
    comment_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de commentaires"))
    rating_score = models.DecimalField(max_digits=3, decimal_places=2, default=Decimal('0.00'),
                                       validators=[MinValueValidator(0), MaxValueValidator(5)],
                                       verbose_name=_("Score de notation"))
    rating_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de notations"))

    # Métadonnées et configuration
    metadata = models.JSONField(default=dict, blank=True, verbose_name=_("Métadonnées"))
    extraction_metadata = models.JSONField(default=dict, blank=True, verbose_name=_("Métadonnées d'extraction"))
    search_keywords = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                                 verbose_name=_("Mots-clés de recherche"))
    auto_generated_tags = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                                     verbose_name=_("Tags générés automatiquement"))

    # Configuration d'accès et fonctionnalités
    is_featured = models.BooleanField(default=False, verbose_name=_("Document vedette"))
    is_searchable = models.BooleanField(default=True, verbose_name=_("Recherchable"))
    is_downloadable = models.BooleanField(default=True, verbose_name=_("Téléchargeable"))
    is_shareable = models.BooleanField(default=True, verbose_name=_("Partageable"))
    is_commentable = models.BooleanField(default=True, verbose_name=_("Commentable"))
    is_ratable = models.BooleanField(default=True, verbose_name=_("Notifiable"))
    requires_approval = models.BooleanField(default=False, verbose_name=_("Nécessite approbation"))
    allow_anonymous_access = models.BooleanField(default=False, verbose_name=_("Accès anonyme autorisé"))

    # Versioning et historique
    version = models.CharField(max_length=20, default='1.0', verbose_name=_("Version"))
    parent_document = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True,
                                        related_name='versions', verbose_name=_("Document parent"))
    is_latest_version = models.BooleanField(default=True, verbose_name=_("Dernière version"))
    version_notes = models.TextField(blank=True, verbose_name=_("Notes de version"))

    # Relations avec documents liés
    related_documents = models.ManyToManyField('self', blank=True, symmetrical=True,
                                               verbose_name=_("Documents liés"))
    supersedes = models.ManyToManyField('self', blank=True, symmetrical=False,
                                        related_name='superseded_by', verbose_name=_("Remplace"))

    # Index de recherche full-text
    search_vector = SearchVectorField(null=True, blank=True)

    # Traçabilité et audit
    processing_log = models.JSONField(default=list, blank=True, verbose_name=_("Journal de traitement"))
    quality_score = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True,
                                        validators=[MinValueValidator(0), MaxValueValidator(1)],
                                        verbose_name=_("Score de qualité"))

    class Meta:
        verbose_name = _("Document")
        verbose_name_plural = _("Documents")
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['status', 'visibility']),
            models.Index(fields=['theme', 'category']),
            models.Index(fields=['uploaded_by', 'created_at']),
            models.Index(fields=['language', 'is_searchable']),
            models.Index(fields=['publication_date', 'effective_date']),
            models.Index(fields=['is_featured', 'status']),
            models.Index(fields=['priority', 'status']),
            GinIndex(fields=['search_vector']),
        ]
        permissions = [
            ('can_approve_document', 'Can approve documents'),
            ('can_reject_document', 'Can reject documents'),
            ('can_publish_document', 'Can publish documents'),
            ('can_access_confidential', 'Can access confidential documents'),
        ]

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)

        # Calcul du checksum si nouveau fichier
        if self.file_path and not self.file_checksum:
            self.file_checksum = self._calculate_checksum()

        # Mise à jour de la taille de fichier
        if self.file_path:
            self.file_size = self.file_path.size
            self.file_type = os.path.splitext(self.original_filename)[1][1:].lower()

        super().save(*args, **kwargs)

    def _calculate_checksum(self):
        """Calcule le checksum SHA-256 du fichier"""
        hash_sha256 = hashlib.sha256()
        try:
            self.file_path.open('rb')
            for chunk in iter(lambda: self.file_path.read(4096), b""):
                hash_sha256.update(chunk)
            self.file_path.close()
            return hash_sha256.hexdigest()
        except:
            return ""

    def get_absolute_url(self):
        return reverse('document-detail', kwargs={'slug': self.slug})

    def can_be_viewed_by(self, user):
        """Vérifie si un utilisateur peut consulter ce document"""
        if self.visibility == self.VisibilityChoices.PUBLIC:
            return True
        if user.is_anonymous:
            return self.allow_anonymous_access
        if user == self.uploaded_by or user.is_staff:
            return True
        if self.visibility == self.VisibilityChoices.PRIVATE:
            return user in self.authorized_users.all()
        return True

    def can_be_downloaded_by(self, user):
        """Vérifie si un utilisateur peut télécharger ce document"""
        return self.is_downloadable and self.can_be_viewed_by(user)

    def can_be_shared_by(self, user):
        """Vérifie si un utilisateur peut partager ce document"""
        return self.is_shareable and self.can_be_viewed_by(user)

    @property
    def file_size_mb(self):
        """Retourne la taille du fichier en MB"""
        return round(self.file_size / (1024 * 1024), 2) if self.file_size else 0

    @property
    def is_expired(self):
        """Vérifie si le document a expiré"""
        if self.expiration_date:
            return timezone.now().date() > self.expiration_date
        return False

    @property
    def needs_review(self):
        """Vérifie si le document nécessite une révision"""
        if self.review_date:
            return timezone.now().date() >= self.review_date
        return False


class DocumentContent(TimeStampedModel):
    """Contenu extrait et structuré des documents"""

    class ContentType(models.TextChoices):
        RAW_TEXT = 'raw_text', _('Texte brut')
        STRUCTURED = 'structured', _('Structuré')
        HTML = 'html', _('HTML')
        MARKDOWN = 'markdown', _('Markdown')
        JSON = 'json', _('JSON')
        XML = 'xml', _('XML')

    class ExtractionMethod(models.TextChoices):
        AUTOMATIC = 'automatic', _('Automatique')
        MANUAL = 'manual', _('Manuel')
        OCR = 'ocr', _('OCR')
        HYBRID = 'hybrid', _('Hybride')
        AI_ENHANCED = 'ai_enhanced', _('Amélioré par IA')

    class ProcessingStatus(models.TextChoices):
        PENDING = 'pending', _('En attente')
        PROCESSING = 'processing', _('En cours')
        COMPLETED = 'completed', _('Terminé')
        FAILED = 'failed', _('Échec')
        NEEDS_REVIEW = 'needs_review', _('Nécessite révision')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='content',
                                    verbose_name=_("Document"))

    # Contenu principal
    raw_content = models.TextField(verbose_name=_("Contenu brut"))
    structured_content = models.JSONField(default=dict, verbose_name=_("Contenu structuré"))
    html_content = models.TextField(blank=True, verbose_name=_("Contenu HTML"))
    markdown_content = models.TextField(blank=True, verbose_name=_("Contenu Markdown"))
    clean_content = models.TextField(blank=True, verbose_name=_("Contenu nettoyé"))

    # Métadonnées d'extraction
    content_type = models.CharField(max_length=20, choices=ContentType.choices,
                                    default=ContentType.RAW_TEXT, verbose_name=_("Type de contenu"))
    extraction_method = models.CharField(max_length=20, choices=ExtractionMethod.choices,
                                         default=ExtractionMethod.AUTOMATIC, verbose_name=_("Méthode d'extraction"))
    processing_status = models.CharField(max_length=20, choices=ProcessingStatus.choices,
                                         default=ProcessingStatus.PENDING, verbose_name=_("Statut de traitement"))
    extraction_confidence = models.DecimalField(max_digits=5, decimal_places=4, default=Decimal('0.0000'),
                                                validators=[MinValueValidator(0), MaxValueValidator(1)],
                                                verbose_name=_("Confiance d'extraction"))

    # Statistiques de contenu
    word_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de mots"))
    character_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de caractères"))
    sentence_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de phrases"))
    paragraph_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de paragraphes"))
    page_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de pages"))
    table_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de tableaux"))
    image_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre d'images"))

    # Enrichissement par IA et NLP
    entities_extracted = models.JSONField(default=dict, blank=True, verbose_name=_("Entités extraites"))
    keywords_extracted = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                                    verbose_name=_("Mots-clés extraits"))
    topics_detected = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                                 verbose_name=_("Sujets détectés"))
    sentiment_score = models.DecimalField(max_digits=3, decimal_places=2, null=True, blank=True,
                                          validators=[MinValueValidator(-1), MaxValueValidator(1)],
                                          verbose_name=_("Score de sentiment"))
    readability_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True,
                                            verbose_name=_("Score de lisibilité"))
    complexity_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True,
                                           verbose_name=_("Score de complexité"))

    # Traitement et performance
    processed_at = models.DateTimeField(null=True, blank=True, verbose_name=_("Traité le"))
    processing_duration = models.DurationField(null=True, blank=True, verbose_name=_("Durée de traitement"))
    processing_errors = models.JSONField(default=list, blank=True, verbose_name=_("Erreurs de traitement"))
    quality_checks = models.JSONField(default=dict, blank=True, verbose_name=_("Vérifications qualité"))

    # Métadonnées techniques
    encoding_detected = models.CharField(max_length=50, blank=True, verbose_name=_("Encodage détecté"))
    language_detected = models.CharField(max_length=10, blank=True, verbose_name=_("Langue détectée"))
    language_confidence = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True,
                                              verbose_name=_("Confiance détection langue"))

    class Meta:
        verbose_name = _("Contenu de document")
        verbose_name_plural = _("Contenus de documents")
        indexes = [
            models.Index(fields=['processing_status']),
            models.Index(fields=['extraction_method', 'processing_status']),
        ]

    def __str__(self):
        return f"Contenu de {self.document.title}"

    def calculate_statistics(self):
        """Calcule automatiquement les statistiques de contenu"""
        if self.clean_content:
            import re
            text = self.clean_content

            # Compte des mots
            words = re.findall(r'\b\w+\b', text)
            self.word_count = len(words)

            # Compte des caractères
            self.character_count = len(text)

            # Compte des phrases
            sentences = re.split(r'[.!?]+', text)
            self.sentence_count = len([s for s in sentences if s.strip()])

            # Compte des paragraphes
            paragraphs = text.split('\n\n')
            self.paragraph_count = len([p for p in paragraphs if p.strip()])

    def get_summary(self, max_words=100):
        """Retourne un résumé du contenu"""
        if self.clean_content:
            words = self.clean_content.split()[:max_words]
            return ' '.join(words) + ('...' if len(words) == max_words else '')
        return ""


class Topic(TimeStampedModel, SoftDeleteModel):
    """Sujets/Topics dans les documents"""

    class TopicType(models.TextChoices):
        CHAPTER = 'chapter', _('Chapitre')
        SECTION = 'section', _('Section')
        SUBSECTION = 'subsection', _('Sous-section')
        ARTICLE = 'article', _('Article')
        PARAGRAPH = 'paragraph', _('Paragraphe')
        APPENDIX = 'appendix', _('Annexe')
        INTRODUCTION = 'introduction', _('Introduction')
        CONCLUSION = 'conclusion', _('Conclusion')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='topics',
                                 verbose_name=_("Document"))
    parent_topic = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True,
                                     related_name='subtopics', verbose_name=_("Topic parent"))

    # Informations principales
    title = models.CharField(max_length=500, verbose_name=_("Titre"))
    slug = models.SlugField(max_length=520, verbose_name=_("Slug"))
    content = models.TextField(verbose_name=_("Contenu"))
    summary = models.TextField(blank=True, verbose_name=_("Résumé"))
    topic_type = models.CharField(max_length=20, choices=TopicType.choices,
                                  default=TopicType.SECTION, verbose_name=_("Type de sujet"))

    order_index = models.PositiveIntegerField(default=0, verbose_name=_("Ordre"))
    level = models.PositiveIntegerField(default=1, verbose_name=_("Niveau hiérarchique"))
    numbering = models.CharField(max_length=50, blank=True, verbose_name=_("Numérotation"))

    # Positionnement dans le document
    start_page = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Page de début"))
    end_page = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Page de fin"))
    start_position = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Position de début"))
    end_position = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Position de fin"))

    metadata = models.JSONField(default=dict, blank=True, verbose_name=_("Métadonnées"))
    keywords = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                          verbose_name=_("Mots-clés"))
    entities = models.JSONField(default=dict, blank=True, verbose_name=_("Entités"))
    concepts = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                          verbose_name=_("Concepts"))

    # Statistiques
    word_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de mots"))
    reading_time_minutes = models.PositiveIntegerField(default=0, verbose_name=_("Temps de lecture (min)"))
    importance_score = models.DecimalField(max_digits=5, decimal_places=4, default=Decimal('0.0000'),
                                           validators=[MinValueValidator(0), MaxValueValidator(1)],
                                           verbose_name=_("Score d'importance"))

    is_highlighted = models.BooleanField(default=False, verbose_name=_("Mis en évidence"))
    is_key_section = models.BooleanField(default=False, verbose_name=_("Section clé"))
    requires_attention = models.BooleanField(default=False, verbose_name=_("Nécessite attention"))

    class Meta:
        verbose_name = _("Sujet")
        verbose_name_plural = _("Sujets")
        ordering = ['document', 'order_index']
        indexes = [
            models.Index(fields=['document', 'order_index']),
            models.Index(fields=['parent_topic', 'order_index']),
            models.Index(fields=['slug']),
            models.Index(fields=['topic_type', 'level']),
        ]
        unique_together = ['document', 'slug']

    def __str__(self):
        return f"{self.document.title} - {self.title}"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)

        # Calcul du niveau hiérarchique
        if self.parent_topic:
            self.level = self.parent_topic.level + 1
        else:
            self.level = 1

        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('topic-detail', kwargs={'document_slug': self.document.slug, 'slug': self.slug})

    def get_children(self):
        """Retourne les sous-sujets directs"""
        return self.subtopics.filter(is_deleted=False).order_by('order_index')

    def get_all_descendants(self):
        """Retourne tous les descendants"""
        descendants = []
        for child in self.get_children():
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants


class Section(TimeStampedModel, SoftDeleteModel):
    """Sections dans les topics"""

    class SectionType(models.TextChoices):
        HEADING = 'heading', _('Titre')
        PARAGRAPH = 'paragraph', _('Paragraphe')
        LIST = 'list', _('Liste')
        TABLE = 'table', _('Tableau')
        IMAGE = 'image', _('Image')
        QUOTE = 'quote', _('Citation')
        CODE = 'code', _('Code')
        FOOTNOTE = 'footnote', _('Note de bas de page')
        FORMULA = 'formula', _('Formule')
        DIAGRAM = 'diagram', _('Diagramme')

    class AlignmentChoices(models.TextChoices):
        LEFT = 'left', _('Gauche')
        CENTER = 'center', _('Centre')
        RIGHT = 'right', _('Droite')
        JUSTIFY = 'justify', _('Justifié')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='sections',
                              verbose_name=_("Sujet"))

    # Contenu principal
    title = models.CharField(max_length=500, blank=True, verbose_name=_("Titre"))
    subtitle = models.CharField(max_length=500, blank=True, verbose_name=_("Sous-titre"))
    content = models.TextField(verbose_name=_("Contenu"))
    raw_content = models.TextField(blank=True, verbose_name=_("Contenu brut"))

    # Type et organisation
    section_type = models.CharField(max_length=20, choices=SectionType.choices,
                                    default=SectionType.PARAGRAPH, verbose_name=_("Type de section"))
    order_index = models.PositiveIntegerField(default=0, verbose_name=_("Ordre"))

    # Formatage et présentation
    formatting = models.JSONField(default=dict, blank=True, verbose_name=_("Formatage"))
    css_classes = ArrayField(models.CharField(max_length=50), default=list, blank=True,
                             verbose_name=_("Classes CSS"))
    alignment = models.CharField(max_length=20, choices=AlignmentChoices.choices,
                                 default=AlignmentChoices.LEFT, verbose_name=_("Alignement"))
    font_size = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Taille de police"))
    font_weight = models.CharField(max_length=20, blank=True, verbose_name=_("Graisse de police"))

    # Positionnement dans le document
    start_page = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Page de début"))
    end_page = models.PositiveIntegerField(null=True, blank=True, verbose_name=_("Page de fin"))
    bbox_coordinates = models.JSONField(default=dict, blank=True, verbose_name=_("Coordonnées bounding box"))

    # Métadonnées
    word_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de mots"))
    character_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de caractères"))

    # Configuration
    is_highlighted = models.BooleanField(default=False, verbose_name=_("Mis en évidence"))
    is_critical = models.BooleanField(default=False, verbose_name=_("Critique"))
    requires_translation = models.BooleanField(default=False, verbose_name=_("Nécessite traduction"))

    class Meta:
        verbose_name = _("Section")
        verbose_name_plural = _("Sections")
        ordering = ['topic', 'order_index']
        indexes = [
            models.Index(fields=['topic', 'order_index']),
            models.Index(fields=['section_type']),
            models.Index(fields=['start_page', 'end_page']),
        ]

    def __str__(self):
        title_part = self.title or f"Section {self.order_index}"
        return f"{self.topic.title} - {title_part}"

    def save(self, *args, **kwargs):
        # Calcul automatique des statistiques
        if self.content:
            import re
            words = re.findall(r'\b\w+\b', self.content)
            self.word_count = len(words)
            self.character_count = len(self.content)

        super().save(*args, **kwargs)


class Paragraph(TimeStampedModel):
    """Paragraphes dans les sections"""

    class ParagraphType(models.TextChoices):
        NORMAL = 'normal', _('Normal')
        INTRODUCTION = 'introduction', _('Introduction')
        CONCLUSION = 'conclusion', _('Conclusion')
        QUOTE = 'quote', _('Citation')
        NOTE = 'note', _('Note')
        WARNING = 'warning', _('Avertissement')
        EXAMPLE = 'example', _('Exemple')
        DEFINITION = 'definition', _('Définition')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    section = models.ForeignKey(Section, on_delete=models.CASCADE, related_name='paragraphs',
                                verbose_name=_("Section"))

    # Contenu
    content = models.TextField(verbose_name=_("Contenu"))
    original_content = models.TextField(blank=True, verbose_name=_("Contenu original"))
    paragraph_type = models.CharField(max_length=20, choices=ParagraphType.choices,
                                      default=ParagraphType.NORMAL, verbose_name=_("Type de paragraphe"))
    order_index = models.PositiveIntegerField(default=0, verbose_name=_("Ordre"))

    # Formatage et style
    formatting = models.JSONField(default=dict, blank=True, verbose_name=_("Formatage"))
    alignment = models.CharField(max_length=20, default='left',
                                 choices=[('left', 'Gauche'), ('center', 'Centre'),
                                          ('right', 'Droite'), ('justify', 'Justifié')],
                                 verbose_name=_("Alignement"))
    indentation = models.PositiveIntegerField(default=0, verbose_name=_("Indentation"))

    # Enrichissement sémantique
    entities = models.JSONField(default=dict, blank=True, verbose_name=_("Entités"))
    keywords = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                          verbose_name=_("Mots-clés"))
    concepts = ArrayField(models.CharField(max_length=100), default=list, blank=True,
                          verbose_name=_("Concepts"))
    named_entities = models.JSONField(default=dict, blank=True, verbose_name=_("Entités nommées"))

    # Statistiques
    word_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de mots"))
    sentence_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de phrases"))
    readability_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True,
                                            verbose_name=_("Score de lisibilité"))

    # Configuration
    is_key_paragraph = models.BooleanField(default=False, verbose_name=_("Paragraphe clé"))
    requires_review = models.BooleanField(default=False, verbose_name=_("Nécessite révision"))

    class Meta:
        verbose_name = _("Paragraphe")
        verbose_name_plural = _("Paragraphes")
        ordering = ['section', 'order_index']
        indexes = [
            models.Index(fields=['section', 'order_index']),
            models.Index(fields=['paragraph_type']),
        ]

    def __str__(self):
        return f"Paragraphe {self.order_index} - {self.section.topic.title}"

    def save(self, *args, **kwargs):
        # Calcul automatique des statistiques
        if self.content:
            import re
            words = re.findall(r'\b\w+\b', self.content)
            self.word_count = len(words)

            sentences = re.split(r'[.!?]+', self.content)
            self.sentence_count = len([s for s in sentences if s.strip()])

        super().save(*args, **kwargs)


class Table(TimeStampedModel):
    """Tableaux extraits des documents"""

    class TableType(models.TextChoices):
        DATA = 'data', _('Données')
        FINANCIAL = 'financial', _('Financier')
        COMPARISON = 'comparison', _('Comparaison')
        SCHEDULE = 'schedule', _('Planning')
        REFERENCE = 'reference', _('Référence')
        MATRIX = 'matrix', _('Matrice')
        SUMMARY = 'summary', _('Résumé')

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    section = models.ForeignKey(Section, on_delete=models.CASCADE, related_name='tables',
                                verbose_name=_("Section"))

    title = models.CharField(max_length=500, blank=True, verbose_name=_("Titre"))
    caption = models.TextField(blank=True, verbose_name=_("Légende"))
    table_type = models.CharField(max_length=20, choices=TableType.choices,
                                  default=TableType.DATA, verbose_name=_("Type de tableau"))

    headers = ArrayField(models.CharField(max_length=200), default=list, verbose_name=_("En-têtes"))
    data = models.JSONField(default=dict, verbose_name=_("Données"))
    raw_data = models.JSONField(default=dict, blank=True, verbose_name=_("Données brutes"))
    processed_data = models.JSONField(default=dict, blank=True, verbose_name=_("Données traitées"))

    row_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de lignes"))
    column_count = models.PositiveIntegerField(default=0, verbose_name=_("Nombre de colonnes"))

    # Organisation
    order_index = models.PositiveIntegerField(default=0, verbose_name=_("Ordre"))

    # Métadonnées d'extraction
    extraction_confidence = models.DecimalField(max_digits=5, decimal_places=4, default=Decimal('0.0000'),
                                                verbose_name=_("Confiance d'extraction"))
    extraction_method = models.CharField(max_length=50, blank=True, verbose_name=_("Méthode d'extraction"))
    bbox_coordinates = models.JSONField(default=dict, blank=True, verbose_name=_("Coordonnées"))

    # Formatage
    styling = models.JSONField(default=dict, blank=True, verbose_name=_("Style"))
    borders = models.JSONField(default=dict, blank=True, verbose_name=_("Bordures"))
    column_widths = ArrayField(models.PositiveIntegerField(), default=list, blank=True,
                               verbose_name=_("Largeurs de colonnes"))

    # Configuration
    has_header_row = models.BooleanField(default=True, verbose_name=_("Ligne d'en-tête"))
    has_totals_row = models.BooleanField(default=False, verbose_name=_("Ligne de totaux"))
    is_transposed = models.BooleanField(default=False, verbose_name=_("Transposé"))
    is_complex = models.BooleanField(default=False, verbose_name=_("Complexe"))

    class Meta:
        verbose_name = _("Tableau")
        verbose_name_plural = _("Tableaux")
        ordering = ['order_index']
        indexes = [
            models.Index(fields=['order_index']),
            models.Index(fields=['table_type']),
        ]

    def __str__(self):
        title_part = self.title or f"Tableau {self.order_index}"
        return f"{self.section.topic.title} - {title_part}"