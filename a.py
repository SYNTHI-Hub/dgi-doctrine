import os
import time
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.storage import Storage
from appwrite.id import ID
from appwrite.query import Query
from datetime import datetime

# Initialize the Appwrite client
client = Client()
client.set_endpoint(os.getenv('NEXT_PUBLIC_APPWRITE_ENDPOINT', 'https://syd.cloud.appwrite.io/v1'))
client.set_project(os.getenv('NEXT_PUBLIC_APPWRITE_PROJECT_ID', '6882384d00032ba96226'))
client.set_key('standard_09172d050204786418044d916a1bfb5b308036025ab4138c577c9d67cc3abebbb6b11d11fda92156aaced3503345eb2482e23873a773107e7a8dd62357dcb9a10048e5d0ef5d183c347cb6ed5d5833960c9bc08a32357e0f1f8690f67f5e24a6cda92f7c51aa2a483370851ccafb51a3267ccf63ff3e4bc66a437869d1a9e5af')  # Set API key for authentication

databases = Databases(client)
storage = Storage(client)

# Database and collection IDs
DATABASE_ID = os.getenv('NEXT_PUBLIC_DATABASE_ID', '68823942003cec72f927')
COLLECTION_ID_SOLUTIONS = os.getenv('NEXT_PUBLIC_COLLECTION_ID_SOLUTIONS', '6882399700312fffced9')
COLLECTION_ID_TECHNOLOGIES = 'technologies'
COLLECTION_ID_USE_CASES = 'use_cases'
COLLECTION_ID_OBJECTIFS = 'objectifs'
COLLECTION_ID_AUTEURS = 'auteurs'

def create_database():
    """Create the database if it doesn't exist."""
    try:
        databases.create(DATABASE_ID, name='Solutions Database')
        print(f"Database {DATABASE_ID} created successfully.")
    except Exception as e:
        if 'already exists' in str(e).lower():
            print(f"Database {DATABASE_ID} already exists.")
        else:
            print(f"Error creating database: {e}")
            raise

def create_collection(collection_id, collection_name, attributes):
    """Create a collection with the specified attributes."""
    try:
        databases.create_collection(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            name=collection_name
        )
        print(f"Collection {collection_name} ({collection_id}) created successfully.")

        # Create attributes for the collection
        for attr in attributes:
            databases.create_ip_attribute(
                database_id=DATABASE_ID,
                collection_id=collection_id,
                **attr
            )
            print(f"Attribute {attr['key']} created for collection {collection_name}.")
            time.sleep(1)  # Add delay to ensure attribute is fully created
    except Exception as e:
        if 'already exists' in str(e).lower():
            print(f"Collection {collection_name} ({collection_id}) already exists.")
        else:
            print(f"Error creating collection {collection_name}: {e}")
            raise

def create_collections():
    """Create all collections with their schemas."""
    # Solutions collection attributes
    solution_attributes = [
        {'key': 'slug', 'type': 'string', 'size': 500, 'required': True},
        {'key': 'titre', 'type': 'string', 'size': 500, 'required': True},
        {'key': 'description_courte', 'type': 'string', 'size': 5000, 'required': False},
        {'key': 'description_longue', 'type': 'string', 'size': 10000, 'required': False},
        {'key': 'images', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'videos_demo', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'technologies_utilisees', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'icon', 'type': 'string', 'size': 500, 'required': False},
        {'key': 'categorie', 'type': 'string', 'size': 100, 'required': False},
        {'key': 'fonctionnalites', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'objectifs', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'use_cases', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'liens_externes', 'type': 'string', 'size': 500, 'required': False, 'array': True},
        {'key': 'auteur', 'type': 'string', 'size': 500, 'required': False},
        {'key': 'date_de_creation', 'type': 'datetime', 'required': True},
        {'key': 'statut', 'type': 'string', 'size': 500, 'required': True},
    ]

    # Technologies collection attributes
    technology_attributes = [
        {'key': 'nom', 'type': 'string', 'size': 500, 'required': True},
        {'key': 'type', 'type': 'string', 'size': 100, 'required': False},
        {'key': 'icone', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'url_doc', 'type': 'string', 'size': 255, 'required': False},
    ]

    # Use Cases collection attributes
    use_case_attributes = [
        {'key': 'titre', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'description', 'type': 'string', 'size': 1000, 'required': False},
        {'key': 'image', 'type': 'string', 'size': 255, 'required': False},
    ]

    # Objectives collection attributes
    objective_attributes = [
        {'key': 'titre', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'description', 'type': 'string', 'size': 1000, 'required': False},
        {'key': 'icon', 'type': 'string', 'size': 255, 'required': False},
    ]

    # Authors collection attributes
    author_attributes = [
        {'key': 'nom', 'type': 'string', 'size': 255, 'required': True},
        {'key': 'poste', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'photo', 'type': 'string', 'size': 255, 'required': False},
        {'key': 'email', 'type': 'string', 'size': 255, 'required': False},
    ]

    # Create all collections
    create_collection(COLLECTION_ID_SOLUTIONS, 'Solutions', solution_attributes)
    create_collection(COLLECTION_ID_TECHNOLOGIES, 'Technologies', technology_attributes)
    create_collection(COLLECTION_ID_USE_CASES, 'Use Cases', use_case_attributes)
    create_collection(COLLECTION_ID_OBJECTIFS, 'Objectifs', objective_attributes)
    create_collection(COLLECTION_ID_AUTEURS, 'Auteurs', author_attributes)

def insert_sample_data():
    """Insert sample data into the collections."""
    try:
        # Insert sample author
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
        print(f"Sample author created with ID: {author_id}")

        # Insert sample technology
        technology_data = {
            'nom': 'Python',
            'type': 'Programming Language',
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
        print(f"Sample technology created with ID: {technology_id}")

        # Insert sample use case
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
        print(f"Sample use case created with ID: {use_case_id}")

        # Insert sample objective
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
        print(f"Sample objective created with ID: {objective_id}")

        # Insert sample solution
        solution_data = {
            'slug': 'sample-solution',
            'titre': 'Sample Solution',
            'description_courte': 'A sample solution for testing.',
            'description_longue': 'This is a detailed description of the sample solution.',
            'images': ['image1-id', 'image2-id'],
            'videos_demo': ['https://example.com/demo.mp4'],
            'technologies_utilisees': [technology_id],
            'icon': 'solution-icon-id',
            'categorie': 'Productivity',
            'fonctionnalites': ['Feature 1', 'Feature 2'],
            'objectifs': [objective_id],
            'use_cases': [use_case_id],
            'liens_externes': ['https://example.com'],
            'auteur': author_id,
            'date_de_creation': datetime.now().isoformat(),
            'statut': 'active'
        }
        solution = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID_SOLUTIONS,
            document_id=ID.unique(),
            data=solution_data
        )
        print(f"Sample solution created with ID: {solution['$id']}")

    except Exception as e:
        print(f"Error inserting sample data: {e}")
        raise

def main():
    """Main function to initialize schema and insert data."""
    try:
        #create_database()
        create_collections()
        time.sleep(2)  # Wait for collections to be fully created
        insert_sample_data()
        print("Database schema and sample data initialized successfully.")
    except Exception as e:
        print(f"Error in initialization: {e}")
        raise

if __name__ == "__main__":
    main()
