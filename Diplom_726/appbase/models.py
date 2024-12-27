from django.db import models
from django.utils.text import slugify


class Buyer(models.Model):
    name = models.CharField(max_length=100, unique=True)
    balance = models.DecimalField(max_digits=10, decimal_places=2)
    age = models.PositiveSmallIntegerField()
    slug = models.SlugField(null=False, unique=True)

    def save(self, *args, **kwargs):
        # if not self.id:
        self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class News(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class MethodML(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    resource = models.URLField()
    slug = models.SlugField(null=False, unique=True)

    def save(self, *args, **kwargs):
        # if not self.id:
        self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class Dataset_Lib(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
