# Generated by Django 5.1.4 on 2024-12-18 23:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('appbase', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='MethodML',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.TextField()),
                ('resource', models.URLField()),
                ('slug', models.SlugField(unique=True)),
            ],
        ),
    ]
