from rest_framework import permissions
from oauth2_provider.contrib.rest_framework import TokenHasScope
from .models import Document, User


def get_document_from_obj(obj):
    """
    Utility function to resolve a Document instance from an object.
    Handles Document, Topic, Section, or related objects.

    Args:
        obj: The object to resolve (e.g., Document, Topic, Section).

    Returns:
        Document instance or None if resolution fails.
    """
    if isinstance(obj, Document):
        return obj
    if hasattr(obj, 'document') and isinstance(obj.document, Document):
        return obj.document
    if hasattr(obj, 'topic') and hasattr(obj.topic, 'document'):
        return obj.topic.document
    if hasattr(obj, 'section') and hasattr(obj.section, 'topic'):
        return obj.section.topic.document
    return None


class CanViewDocument(permissions.BasePermission):
    """Permission pour consulter un document selon sa visibilité et les scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de consulter ce document."

    def has_permission(self, request, view):
        """
        Vérifie si l'utilisateur est authentifié et a le scope 'read' pour la vue.

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut accéder à la vue.
        """
        return (
                request.user
                and request.user.is_authenticated
                and TokenHasScope().has_permission(request, view)
        )

    def has_object_permission(self, request, view, obj):
        """
        Vérifie si l'utilisateur peut consulter l'objet (Document ou lié).

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object (Document, Topic, Section, etc.).

        Returns:
            bool: True si l'utilisateur peut consulter l'objet.
        """
        if not TokenHasScope().has_permission(request, view):
            return False

        document = get_document_from_obj(obj)
        if document is None:
            return False

        try:
            return document.can_be_viewed_by(request.user)
        except AttributeError:
            return False


class CanManageDocument(permissions.BasePermission):
    """Permission pour gérer un document (modifier, supprimer, traiter) avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de gérer ce document."

    def has_permission(self, request, view):
        """
        Vérifie si l'utilisateur est authentifié et a le scope 'write' pour la vue.

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut accéder à la vue.
        """
        return (
                request.user
                and request.user.is_authenticated
                and TokenHasScope().has_permission(request, view)
        )

    def has_object_permission(self, request, view, obj):
        """
        Vérifie si l'utilisateur peut gérer l'objet (Document ou lié).

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object (Document, Topic, Section, etc.).

        Returns:
            bool: True si l'utilisateur peut gérer l'objet.
        """
        if not TokenHasScope().has_permission(request, view):
            return False

        user = request.user
        document = get_document_from_obj(obj)
        if document is None:
            return False

        try:
            # Super utilisateurs
            if user.is_superuser:
                return True

            # Propriétaire du document
            if document.uploaded_by == user:
                return True

            # Administrateurs et modérateurs
            if hasattr(user, 'role') and user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
                return True

            # Gestionnaires du thème
            if hasattr(document, 'theme') and hasattr(document.theme, 'managed_by'):
                return user in document.theme.managed_by.all()

        except AttributeError:
            return False

        return False


class CanProcessDocument(permissions.BasePermission):
    """Permission pour lancer le traitement d'un document avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de traiter ce document."

    def has_permission(self, request, view):
        """
        Vérifie si l'utilisateur peut lancer le traitement (rôles spécifiques et scope 'write').

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut accéder à la vue.
        """
        user = request.user
        return (
                user
                and user.is_authenticated
                and hasattr(user, 'role')
                and user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR, User.RoleChoices.ANALYST]
                and TokenHasScope().has_permission(request, view)
        )

    def has_object_permission(self, request, view, obj):
        """
        Vérifie si l'utilisateur peut traiter l'objet (Document uniquement).

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object (Document).

        Returns:
            bool: True si l'utilisateur peut traiter l'objet.
        """
        if not TokenHasScope().has_permission(request, view):
            return False

        user = request.user
        document = get_document_from_obj(obj)
        if document is None:
            return False

        try:
            # Super utilisateurs
            if user.is_superuser:
                return True

            # Propriétaire du document
            if document.uploaded_by == user:
                return True

            # Administrateurs, modérateurs et analystes
            if hasattr(user, 'role') and user.role in [
                User.RoleChoices.ADMIN,
                User.RoleChoices.MODERATOR,
                User.RoleChoices.ANALYST
            ]:
                return True

        except AttributeError:
            return False

        return False


class CanUploadDocument(permissions.BasePermission):
    """Permission pour uploader des documents avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation d'uploader des documents."

    def has_permission(self, request, view):
        """
        Vérifie si l'utilisateur peut uploader des documents (statut, quota, scope 'write').

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut uploader.
        """
        user = request.user
        if not user or not user.is_authenticated or not TokenHasScope().has_permission(request, view):
            return False

        try:
            # Vérification du statut de l'utilisateur
            if user.status != User.StatusChoices.ACTIVE:
                return False

            # Vérification des quotas
            if hasattr(user, 'can_upload_documents') and not user.can_upload_documents():
                self.message = f"Limite de documents atteinte. Maximum: {getattr(user, 'max_documents_upload', 'inconnu')}"
                return False

            return True
        except AttributeError:
            return False


class IsOwnerOrManagerOrReadOnly(permissions.BasePermission):
    """Permission pour les propriétaires, managers ou lecture seule avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de modifier cet objet."

    def has_permission(self, request, view):
        """
        Vérifie les permissions au niveau de la vue (lecture ou écriture).

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut accéder à la vue.
        """
        if request.method in permissions.SAFE_METHODS:
            return request.user and request.user.is_authenticated and TokenHasScope().has_permission(request, view)
        return request.user and request.user.is_authenticated and TokenHasScope().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        """
        Vérifie les permissions au niveau de l'objet (lecture ou écriture).

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object being accessed.

        Returns:
            bool: True si l'utilisateur a la permission.
        """
        if not TokenHasScope().has_permission(request, view):
            return False

        user = request.user

        # Lecture autorisée pour tous les utilisateurs authentifiés
        if request.method in permissions.SAFE_METHODS:
            return True

        try:
            # Super utilisateurs
            if user.is_superuser:
                return True

            # Propriétaire
            if hasattr(obj, 'uploaded_by') and obj.uploaded_by == user:
                return True

            # Manager de l'utilisateur
            if hasattr(obj, 'uploaded_by') and hasattr(obj.uploaded_by, 'manager') and obj.uploaded_by.manager == user:
                return True

            # Administrateurs et modérateurs
            if hasattr(user, 'role') and user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
                return True

        except AttributeError:
            return False

        return False


class HasAPIAccess(permissions.BasePermission):
    """Permission pour l'accès API avec scopes OAuth."""

    message = "Accès API non autorisé."

    def has_permission(self, request, view):
        """
        Vérifie si l'utilisateur a un accès API activé et les scopes nécessaires.

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur a l'accès API.
        """
        user = request.user
        if not user or not user.is_authenticated or not TokenHasScope().has_permission(request, view):
            return False

        try:
            return user.api_access_enabled
        except AttributeError:
            return False


class DocumentViewPermissions(permissions.BasePermission):
    """Permissions combinées pour la consultation de documents avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de consulter ce document."

    def has_permission(self, request, view):
        """
        Vérifie les permissions de consultation au niveau de la vue.

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut consulter.
        """
        return CanViewDocument().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        """
        Vérifie les permissions de consultation au niveau de l'objet.

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object being accessed.

        Returns:
            bool: True si l'utilisateur peut consulter l'objet.
        """
        return CanViewDocument().has_object_permission(request, view, obj)


class DocumentManagePermissions(permissions.BasePermission):
    """Permissions combinées pour la gestion de documents avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de gérer ce document."

    def has_permission(self, request, view):
        """
        Vérifie les permissions de gestion au niveau de la vue.

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut gérer.
        """
        if request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_permission(request, view)
        return CanManageDocument().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        """
        Vérifie les permissions de gestion au niveau de l'objet.

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object being accessed.

        Returns:
            bool: True si l'utilisateur peut gérer l'objet.
        """
        if request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_object_permission(request, view, obj)
        return CanManageDocument().has_object_permission(request, view, obj)


class DocumentProcessPermissions(permissions.BasePermission):
    """Permissions combinées pour le traitement de documents avec scopes OAuth."""

    message = "Vous n'avez pas l'autorisation de traiter ce document."

    def has_permission(self, request, view):
        """
        Vérifie les permissions de traitement au niveau de la vue.

        Args:
            request: The HTTP request.
            view: The view being accessed.

        Returns:
            bool: True si l'utilisateur peut traiter.
        """
        if view.action in ['process_content', 'upload_and_process']:
            return CanProcessDocument().has_permission(request, view)
        elif request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_permission(request, view)
        return CanManageDocument().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        """
        Vérifie les permissions de traitement au niveau de l'objet.

        Args:
            request: The HTTP request.
            view: The view being accessed.
            obj: The object being accessed.

        Returns:
            bool: True si l'utilisateur peut traiter l'objet.
        """
        if view.action in ['process_content']:
            return CanProcessDocument().has_object_permission(request, view, obj)
        elif request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_object_permission(request, view, obj)
        return CanManageDocument().has_object_permission(request, view, obj)