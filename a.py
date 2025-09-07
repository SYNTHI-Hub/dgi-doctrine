import os
import time
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.storage import Storage
from appwrite.id import ID
from datetime import datetime

# Initialize the Appwrite client
client = Client()
client.set_endpoint(os.getenv('NEXT_PUBLIC_APPWRITE_ENDPOINT', 'https://syd.cloud.appwrite.io/v1'))
client.set_project(os.getenv('NEXT_PUBLIC_APPWRITE_PROJECT_ID', '6882384d00032ba96226'))
client.set_key('standard_09172d050204786418044d916a1bfb5b308036025ab4138c577c9d67cc3abebbb6b11d11fda92156aaced3503345eb2482e23873a773107e7a8dd62357dcb9a10048e5d0ef5d183c347cb6ed5d5833960c9bc08a32357e0f1f8690f67f5e24a6cda92f7c51aa2a483370851ccafb51a3267ccf63ff3e4bc66a437869d1a9e5af')  # Set API key for authentication


databases = Databases(client)
storage = Storage(client)

# IDs - récupérés depuis variables d'environnement ou valeurs par défaut
DATABASE_ID = os.getenv('NEXT_PUBLIC_DATABASE_ID', '68823942003cec72f927')

COLLECTION_ID_SOLUTIONS = os.getenv('NEXT_PUBLIC_COLLECTION_ID_SOLUTIONS', '6895f48b00104b1c6afe')
COLLECTION_ID_TECHNOLOGIES = os.getenv('NEXT_PUBLIC_COLLECTION_ID_TECHNOLOGIES', '6895f4c800177aedc217')
COLLECTION_ID_USE_CASES = os.getenv('NEXT_PUBLIC_COLLECTION_ID_USE_CASES', '6895f4e4003014f7a29d')
COLLECTION_ID_OBJECTIFS = os.getenv('NEXT_PUBLIC_COLLECTION_ID_OBJECTIVES', '6895f509001027c30af4b')  # attention au nom variable
COLLECTION_ID_AUTEURS = os.getenv('NEXT_PUBLIC_COLLECTION_ID_AUTHORS', '6895f5220037445d3b8b')


def create_collection(collection_id, collection_name, attributes):
    try:
        databases.create_collection(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            name=collection_name
        )
        print(f"Collection {collection_name} ({collection_id}) créée avec succès.")
    except Exception as e:
        if 'already exists' in str(e).lower():
            print(f"Collection {collection_name} ({collection_id}) existe déjà.")
        else:
            print(f"Erreur lors de la création de la collection {collection_name} : {e}")
            raise

    # Création des attributs (champs)
    for attr in attributes:
        try:
            # Utilisation de la bonne méthode selon le type
            if attr['type'] == 'string':
                databases.create_string_attribute(
                    database_id=DATABASE_ID,
                    collection_id=collection_id,
                    key=attr['key'],
                    size=attr.get('size', 255),
                    required=attr.get('required', False),
                    array=attr.get('array', False)
                )
            elif attr['type'] == 'datetime':
                databases.create_datetime_attribute(
                    database_id=DATABASE_ID,
                    collection_id=collection_id,
                    key=attr['key'],
                    required=attr.get('required', False)
                )
            else:
                # Ajouter d'autres types si nécessaire
                print(f"Type d'attribut non géré: {attr['type']} pour {attr['key']}")
            print(f"Attribut {attr['key']} créé pour {collection_name}.")
            time.sleep(1)  # éviter de surcharger l'API
        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"Attribut {attr['key']} existe déjà dans {collection_name}.")
            else:
                print(f"Erreur lors de la création de l'attribut {attr['key']} : {e}")
                raise


def create_collections():
    solution_attributes = [
        {'key': 'slug', 'type': 'string', 'size': 500, 'required': True},
        {'key': 'titre', 'type': 'string', 'size': 500, 'required': True},
        {'key': 'description_courte', 'type': 'string', 'size': 500, 'required': True},
        {'key': 'description_longue', 'type': 'string', 'size': 1000, 'required': True},
        {'key': 'images', 'type': 'string', 'size': 500, 'array': True, 'required': False},
        {'key': 'videos_demo', 'type': 'string', 'size': 500, 'array': True, 'required': False},
        {'key': 'technologies_utilisees', 'type': 'string', 'size': 100, 'array': True, 'required': True},  # IDs de technologies
        {'key': 'icon', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'categorie', 'type': 'string', 'size': 100, 'required': True},
        {'key': 'fonctionnalites', 'type': 'string', 'size': 500, 'array': True, 'required': True},
        {'key': 'objectifs', 'type': 'string', 'size': 100, 'array': True, 'required': True},  # IDs d'objectifs
        {'key': 'use_cases', 'type': 'string', 'size': 100, 'array': True, 'required': True},  # IDs de cas d'usage
        {'key': 'liens_externes', 'type': 'string', 'size': 500, 'array': True, 'required': False},
        {'key': 'auteur', 'type': 'string', 'size': 100, 'required': True},  # ID auteur
        {'key': 'date_de_creation', 'type': 'datetime', 'required': True},
        {'key': 'statut', 'type': 'string', 'size': 50, 'required': True},
        {'key': 'tags', 'type': 'string', 'size': 255, 'array': True, 'required': False},
    ]

    technology_attributes = [
        {'key': 'nom', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'type', 'type': 'string', 'size': 50, 'required': True},
        {'key': 'icone', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'url_doc', 'type': 'string', 'size': 255, 'required': False},
    ]

    use_case_attributes = [
        {'key': 'titre', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'description', 'type': 'string', 'size': 250, 'required': False},
        {'key': 'image', 'type': 'string', 'size': 255, 'required': False},
    ]

    objective_attributes = [
        {'key': 'titre', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'description', 'type': 'string', 'size': 250, 'required': False},
        {'key': 'icon', 'type': 'string', 'size': 255, 'required': False},
    ]

    author_attributes = [
        {'key': 'nom', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'poste', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'photo', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'email', 'type': 'string', 'size': 255, 'required': True},
    ]

    create_collection(COLLECTION_ID_SOLUTIONS, 'Solutions', solution_attributes)
    create_collection(COLLECTION_ID_TECHNOLOGIES, 'Technologies', technology_attributes)
    create_collection(COLLECTION_ID_USE_CASES, 'Use Cases', use_case_attributes)
    create_collection(COLLECTION_ID_OBJECTIFS, 'Objectifs', objective_attributes)
    create_collection(COLLECTION_ID_AUTEURS, 'Auteurs', author_attributes)


def insert_sample_data():
    try:
        author_data = {
            'nom': 'John Doe',
            'poste': 'Lead Developer',
            'email': 'john.doe@example.com',
            'photo': 'sample-photo-id'
        }
        author = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID_AUTEURS,
            document_id=ID.unique(),
            data=author_data
        )
        author_id = author['$id']
        print(f"Auteur sample créé avec ID: {author_id}")

        technology_data = {
            'nom': 'Python',
            'type': 'langage',
            'icone': 'python-icon-id',
            'url_doc': 'https://python.org'
        }
        technology = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID_TECHNOLOGIES,
            document_id=ID.unique(),
            data=technology_data
        )
        technology_id = technology['$id']
        print(f"Technologie sample créée avec ID: {technology_id}")

        use_case_data = {
            'titre': 'Data Analysis',
            'description': 'Analyze large datasets efficiently.',
            'image': 'use-case-image-id'
        }
        use_case = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID_USE_CASES,
            document_id=ID.unique(),
            data=use_case_data
        )
        use_case_id = use_case['$id']
        print(f"Cas d’usage sample créé avec ID: {use_case_id}")

        objective_data = {
            'titre': 'Improve Performance',
            'description': 'Enhance system performance and scalability.',
            'icon': 'objective-icon-id'
        }
        objective = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID_OBJECTIFS,
            document_id=ID.unique(),
            data=objective_data
        )
        objective_id = objective['$id']
        print(f"Objectif sample créé avec ID: {objective_id}")

        solution_data = {
            'slug': 'sample-solution',
            'titre': 'Sample Solution',
            'description_courte': 'A sample solution for testing.',
            'description_longue': 'This is a detailed description of the sample solution.',
            'images': ['image1-id', 'image2-id'],
            'videos_demo': ['https://example.com/demo.mp4'],
            'technologies_utilisees': [technology_id],
            'icon': 'solution-icon-id',
            'categorie': 'Automatisation',  # Doit correspondre à l'enum SolutionCategory
            'fonctionnalites': ['Feature 1', 'Feature 2'],
            'objectifs': [objective_id],
            'use_cases': [use_case_id],
            'liens_externes': ['https://example.com'],
            'auteur': author_id,
            'date_de_creation': datetime.utcnow().isoformat(),
            'statut': 'active',
            'tags': ['sample', 'test']
        }
        solution = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID_SOLUTIONS,
            document_id=ID.unique(),
            data=solution_data
        )
        print(f"Solution sample créée avec ID: {solution['$id']}")

    except Exception as e:
        print(f"Erreur lors de l’insertion des données sample: {e}")
        raise


def main():
    try:
        #create_database()  # facultatif si base déjà créée
        create_collections()
        time.sleep(2)
        insert_sample_data()
        print("Initialisation des collections et données sample terminée.")
    except Exception as e:
        print(f"Erreur lors de l'initialisation : {e}")
        raise


if __name__ == "__main__":
    main()
