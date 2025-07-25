from django.contrib.auth.base_user import BaseUserManager
from django.utils.translation import gettext_lazy as _

class CustomUserManager(BaseUserManager):
    """Manager personnalisé pour l'utilisateur basé sur l'email."""

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError(_("L'adresse email est obligatoire."))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(_("Le superutilisateur doit avoir is_staff=True."))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_("Le superutilisateur doit avoir is_superuser=True."))

        return self.create_user(email, password, **extra_fields)
