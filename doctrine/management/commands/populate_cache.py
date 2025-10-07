from django.core.management.base import BaseCommand
from doctrine.views_1 import document_cache


class Command(BaseCommand):
    help = 'Peuple le cache Redis avec les contenus extraits des documents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--page-size',
            type=int,
            default=20,
            help='Taille de page pour la mise en cache (défaut: 20)'
        )
        parser.add_argument(
            '--max-pages',
            type=int,
            default=20,
            help='Nombre maximum de pages à mettre en cache (défaut: 20)'
        )
        parser.add_argument(
            '--invalidate',
            action='store_true',
            help='Invalide le cache existant avant de le repeupler'
        )

    def handle(self, *args, **options):
        page_size = options['page_size']
        max_pages = options['max_pages']
        invalidate = options['invalidate']

        self.stdout.write('Début du peuplement du cache Redis...')

        if invalidate:
            self.stdout.write('Invalidation du cache existant...')
            document_cache.invalidate_cache()

        # Peupler le cache
        document_cache.populate_cache_from_db(page_size=page_size, max_pages=max_pages)

        self.stdout.write(
            self.style.SUCCESS(
                f'Cache peuplé avec succès ! '
                f'(page_size={page_size}, max_pages={max_pages})'
            )
        )