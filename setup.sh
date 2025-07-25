# ===================================
# scripts/setup.sh - Script d'installation et configuration
# ===================================

#!/bin/bash

set -e

echo "🚀 Configuration du projet Document Management"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'affichage avec couleur
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérification des prérequis
check_requirements() {
    print_status "Vérification des prérequis..."

    # Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 n'est pas installé. Veuillez l'installer (version 3.9+ recommandée)."
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    # Utilisation de bc pour la comparaison de versions flottantes
    if (( $(echo "$python_version < 3.9" | bc -l) )); then
        print_error "Python 3.9+ requis, version actuelle: $python_version. Veuillez mettre à jour Python."
        exit 1
    fi

    # PostgreSQL client
    if ! command -v psql &> /dev/null; then
        print_warning "Client PostgreSQL (psql) non trouvé. Assurez-vous que PostgreSQL est installé et dans votre PATH si vous utilisez une base de données PostgreSQL."
    fi

    # Redis client
    if ! command -v redis-cli &> /dev/null; then
        print_warning "Client Redis (redis-cli) non trouvé. Assurez-vous que Redis est installé et en cours d'exécution si vous utilisez Celery ou le cache Redis."
    fi

    print_success "Prérequis vérifiés"
}

# Création de l'environnement virtuel
setup_virtualenv() {
    print_status "Configuration de l'environnement virtuel..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Environnement virtuel créé"
    else
        print_warning "Environnement virtuel 'venv' existe déjà."
    fi

    source venv/bin/activate

    # Mise à jour de pip
    print_status "Mise à jour de pip, setuptools et wheel..."
    pip install --upgrade pip setuptools wheel
    print_success "Pip, setuptools et wheel mis à jour"
}

# Installation des dépendances
install_dependencies() {
    print_status "Installation des dépendances Python..."

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dépendances installées depuis requirements.txt"
    else
        print_error "Fichier 'requirements.txt' non trouvé. Impossible d'installer les dépendances."
        exit 1
    fi
}

# Configuration de l'environnement
setup_environment() {
    print_status "Configuration des variables d'environnement (.env)..."

    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Fichier .env créé à partir de .env.example."
            print_warning "Veuillez modifier le fichier .env avec vos configurations spécifiques (base de données, etc.)."
        else
            print_error "Fichier .env.example non trouvé. Impossible de créer le fichier .env."
            exit 1
        fi
    else
        print_warning "Fichier .env existe déjà."
    fi

    # Génération d'une clé secrète si nécessaire
    # Utilise 'sed -i '' ' pour la compatibilité macOS
    if ! grep -q "SECRET_KEY=" .env || grep -q "your-secret-key-here" .env; then
        print_status "Génération d'une nouvelle clé secrète Django..."
        secret_key=$(python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
        # Utilisation de 'sed -i '' ' pour la compatibilité macOS
        sed -i '' "s/SECRET_KEY=.*/SECRET_KEY='$secret_key'/" .env
        print_success "Clé secrète générée et ajoutée à .env"
    fi
}

# Création des répertoires nécessaires
create_directories() {
    print_status "Création des répertoires du projet..."

    directories=("logs" "media" "media/documents" "staticfiles" "tmp/celery")

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Répertoire '$dir' créé"
        else
            print_warning "Répertoire '$dir' existe déjà"
        fi
    done
}

# Configuration de la base de données
setup_database() {
    print_status "Configuration de la base de données..."

    # Chargement des variables d'environnement pour cette session de script
    # Note: Cela ne les rend pas persistantes dans votre shell principal.
    if [ -f ".env" ]; then
        source .env
    else
        print_error "Le fichier .env est manquant. Impossible de configurer la base de données."
        exit 1
    fi

    # Test de connexion à PostgreSQL si psql est disponible et DB_ENGINE est défini
    if [[ "$DB_ENGINE" == *"postgresql"* ]] && command -v psql &> /dev/null; then
        print_status "Tentative de connexion à PostgreSQL..."
        if PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "SELECT 1;" &> /dev/null; then
            print_success "Connexion à PostgreSQL réussie."

            print_status "Vérification et création de la base de données '$DB_NAME' si elle n'existe pas..."
            PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\";" 2>/dev/null || print_warning "Base de données '$DB_NAME' existe déjà ou une erreur est survenue lors de la création."
        else
            print_error "Impossible de se connecter à PostgreSQL avec les informations fournies dans .env."
            print_warning "Vérifiez vos paramètres de base de données (DB_HOST, DB_USER, DB_PASSWORD) et assurez-vous que le serveur PostgreSQL est en cours d'exécution."
            # Ne pas exit ici, car l'utilisateur pourrait utiliser SQLite
        fi
    elif [[ "$DB_ENGINE" == *"sqlite3"* ]]; then
        print_status "Utilisation de SQLite. La base de données sera créée automatiquement par Django."
    else
        print_warning "Type de base de données non reconnu ou client PostgreSQL non disponible. Les migrations seront tentées mais la connexion à la DB pourrait échouer."
    fi

    # Exécution des migrations Django
    print_status "Exécution des migrations Django..."
    python manage.py makemigrations # Génère les migrations pour toutes les applications
    python manage.py migrate
    print_success "Migrations appliquées"
}

# Collecte des fichiers statiques
collect_static() {
    print_status "Collecte des fichiers statiques..."
    python manage.py collectstatic --noinput
    print_success "Fichiers statiques collectés"
}

# Création d'un superutilisateur
create_superuser() {
    print_status "Création d'un superutilisateur..."

    echo "Souhaitez-vous créer un superutilisateur Django maintenant? (o/n)"
    read -r response

    if [[ "$response" =~ ^[Oo]$ ]]; then
        python manage.py createsuperuser
        print_success "Superutilisateur créé"
    else
        print_warning "Vous pouvez créer un superutilisateur plus tard avec: 'python manage.py createsuperuser'"
    fi
}

# Test de l'installation
test_installation() {
    print_status "Test de l'installation..."

    # Test des imports Python et de la configuration de Django
    python -c "
import django
import os
import sys

# Assurez-vous que le chemin du projet est dans PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

try:
    django.setup()
    from doctrine.models import User, Document, Theme
    # from doctrine.services.document_processor import document_processor # Décommentez si ce service existe et est importable
    print('✓ Imports Django et modèles réussis.')
except Exception as e:
    print(f'✗ Erreur lors des imports Django ou des modèles: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        print_success "Tests d'imports Django réussis."
    else
        print_error "Erreur lors des tests d'imports Django. Vérifiez votre configuration et vos modèles."
        exit 1
    fi

    # Test de la connexion Redis si client est disponible
    if command -v redis-cli &> /dev/null; then
        print_status "Test de la connexion Redis..."
        if redis-cli ping &> /dev/null; then
            print_success "Connexion Redis réussie."
        else
            print_warning "Redis non accessible ou non démarré. Assurez-vous que Redis est en cours d'exécution si vous utilisez Celery ou le cache Redis."
        fi
    else
        print_warning "Client Redis (redis-cli) non trouvé. Impossible de tester la connexion Redis."
    fi
}

# Fonction principale
main() {
    echo "======================================"
    echo "   Document Management Setup"
    echo "======================================"

    check_requirements
    setup_virtualenv
    install_dependencies
    setup_environment
    create_directories
    setup_database
    collect_static
    create_superuser
    test_installation

    echo ""
    print_success "🎉 Installation terminée avec succès!"
    echo ""
    echo "Étapes suivantes:"
    echo "1. Activez l'environnement virtuel: source venv/bin/activate"
    echo "2. Configurez vos paramètres dans .env (si ce n'est pas déjà fait)"
    echo "3. Démarrez Redis (si utilisé): redis-server"
    echo "4. Démarrez Celery Worker (si utilisé): celery -A core worker -l info" # Assurez-vous que 'core' est le nom de votre module Django principal
    echo "5. Démarrez Celery Beat (si utilisé): celery -A core beat -l info"
    echo "6. Démarrez le serveur Django: python manage.py runserver"
    echo ""
    echo "URLs importantes:"
    echo "- Interface admin: http://localhost:8000/admin/"
    echo "- API docs: http://localhost:8000/api/v1/processing/docs/ (ou votre chemin)"
    echo "- Flower (Celery monitoring, si utilisé): http://localhost:5555/"
    echo ""
    print_status "Pour démarrer tous les services (Redis, Celery, Django), utilisez: ./scripts/start_services.sh"
}

# Exécution du script
main "$@"
