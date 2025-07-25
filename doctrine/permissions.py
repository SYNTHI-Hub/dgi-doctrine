from rest_framework import permissions
from .models import Document, User


class CanViewDocument(permissions.BasePermission):
    """Permission pour voir un document selon sa visibilité"""

    message = "Vous n'avez pas l'autorisation de consulter ce document."

    def has_permission(self, request, view):
        """Permission au niveau de la vue"""
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request, view, obj):
        """Permission au niveau de l'objet"""
        if isinstance(obj, Document):
            return obj.can_be_viewed_by(request.user)

        # Pour les objets liés au document (Topic, Section, etc.)
        if hasattr(obj, 'document'):
            return obj.document.can_be_viewed_by(request.user)
        elif hasattr(obj, 'topic') and hasattr(obj.topic, 'document'):
            return obj.topic.document.can_be_viewed_by(request.user)
        elif hasattr(obj, 'section') and hasattr(obj.section, 'topic'):
            return obj.section.topic.document.can_be_viewed_by(request.user)

        return False


class CanManageDocument(permissions.BasePermission):
    """Permission pour gérer un document (modifier, supprimer, traiter)"""

    message = "Vous n'avez pas l'autorisation de gérer ce document."

    def has_permission(self, request, view):
        """Permission au niveau de la vue"""
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request, view, obj):
        """Permission au niveau de l'objet"""
        user = request.user

        if isinstance(obj, Document):
            document = obj
        elif hasattr(obj, 'document'):
            document = obj.document
        elif hasattr(obj, 'topic') and hasattr(obj.topic, 'document'):
            document = obj.topic.document
        elif hasattr(obj, 'section') and hasattr(obj.section, 'topic'):
            document = obj.section.topic.document
        else:
            return False

        # Super utilisateurs
        if user.is_superuser:
            return True

        # Propriétaire du document
        if document.uploaded_by == user:
            return True

        # Administrateurs et modérateurs
        if user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return True

        # Gestionnaires du thème
        if hasattr(document.theme, 'managed_by') and user in document.theme.managed_by.all():
            return True

        return False


class CanProcessDocument(permissions.BasePermission):
    """Permission pour lancer le traitement d'un document"""

    message = "Vous n'avez pas l'autorisation de traiter ce document."

    def has_permission(self, request, view):
        """Permission au niveau de la vue"""
        user = request.user
        return (user and user.is_authenticated and
                user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR, User.RoleChoices.ANALYST])

    def has_object_permission(self, request, view, obj):
        """Permission au niveau de l'objet"""
        user = request.user

        if isinstance(obj, Document):
            document = obj
        else:
            return False

        # Super utilisateurs
        if user.is_superuser:
            return True

        # Propriétaire du document
        if document.uploaded_by == user:
            return True

        # Administrateurs, modérateurs et analystes
        if user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR, User.RoleChoices.ANALYST]:
            return True

        return False


class CanUploadDocument(permissions.BasePermission):
    """Permission pour uploader des documents"""

    message = "Vous n'avez pas l'autorisation d'uploader des documents."

    def has_permission(self, request, view):
        """Permission au niveau de la vue"""
        user = request.user

        if not user or not user.is_authenticated:
            return False

        # Vérification du statut de l'utilisateur
        if user.status != User.StatusChoices.ACTIVE:
            return False

        # Vérification des quotas
        if not user.can_upload_documents():
            self.message = f"Limite de documents atteinte. Maximum: {user.max_documents_upload}"
            return False

        return True


class IsOwnerOrManagerOrReadOnly(permissions.BasePermission):
    """Permission pour les propriétaires, managers ou lecture seule"""

    def has_permission(self, request, view):
        """Permission au niveau de la vue"""
        if request.method in permissions.SAFE_METHODS:
            return request.user and request.user.is_authenticated

        return request.user and request.user.is_authenticated

    def has_object_permission(self, request, view, obj):
        """Permission au niveau de l'objet"""
        # Lecture autorisée pour tous les utilisateurs authentifiés
        if request.method in permissions.SAFE_METHODS:
            return True

        user = request.user

        # Super utilisateurs
        if user.is_superuser:
            return True

        # Propriétaire
        if hasattr(obj, 'uploaded_by') and obj.uploaded_by == user:
            return True

        # Manager de l'utilisateur
        if hasattr(obj, 'uploaded_by') and obj.uploaded_by.manager == user:
            return True

        # Administrateurs et modérateurs
        if user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return True

        return False


class HasAPIAccess(permissions.BasePermission):
    """Permission pour l'accès API"""

    message = "Accès API non autorisé."

    def has_permission(self, request, view):
        """Permission au niveau de la vue"""
        user = request.user

        if not user or not user.is_authenticated:
            return False

        return user.api_access_enabled


# Permissions combinées pour différents cas d'usage
class DocumentViewPermissions(permissions.BasePermission):
    """Permissions combinées pour la consultation de documents"""

    def has_permission(self, request, view):
        return CanViewDocument().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        return CanViewDocument().has_object_permission(request, view, obj)


class DocumentManagePermissions(permissions.BasePermission):
    """Permissions combinées pour la gestion de documents"""

    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_permission(request, view)
        else:
            return CanManageDocument().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_object_permission(request, view, obj)
        else:
            return CanManageDocument().has_object_permission(request, view, obj)


class DocumentProcessPermissions(permissions.BasePermission):
    """Permissions combinées pour le traitement de documents"""

    def has_permission(self, request, view):
        if view.action in ['process_content', 'upload_and_process']:
            return CanProcessDocument().has_permission(request, view)
        elif request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_permission(request, view)
        else:
            return CanManageDocument().has_permission(request, view)

    def has_object_permission(self, request, view, obj):
        if view.action in ['process_content']:
            return CanProcessDocument().has_object_permission(request, view, obj)
        elif request.method in permissions.SAFE_METHODS:
            return CanViewDocument().has_object_permission(request, view, obj)
        else:
            return CanManageDocument().has_object_permission(request, view, obj)