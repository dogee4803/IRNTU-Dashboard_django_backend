# Generated by Django 5.1.2 on 2024-12-08 12:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0002_rename_nagruzka_complexity_and_more'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='course_project',
            options={'verbose_name': 'Курсовая работа/проект', 'verbose_name_plural': 'Курсовые работы/проекты'},
        ),
        migrations.AlterModelOptions(
            name='hours_per_semestr',
            options={'verbose_name': 'Часы за семестр', 'verbose_name_plural': 'Часы за семестр'},
        ),
        migrations.RemoveField(
            model_name='student',
            name='name',
        ),
        migrations.RemoveField(
            model_name='teacher',
            name='name',
        ),
        migrations.AddField(
            model_name='student',
            name='first_name',
            field=models.TextField(default='', max_length=255, verbose_name='Имя'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='student',
            name='last_name',
            field=models.TextField(default='', max_length=255, verbose_name='Фамилия'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='student',
            name='patronymic',
            field=models.TextField(blank=True, max_length=255, null=True, verbose_name='Отчество'),
        ),
        migrations.AddField(
            model_name='teacher',
            name='first_name',
            field=models.TextField(default='', max_length=255, verbose_name='Имя'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='teacher',
            name='last_name',
            field=models.TextField(default='', max_length=255, verbose_name='Фамилия'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='teacher',
            name='patronymic',
            field=models.TextField(blank=True, max_length=255, null=True, verbose_name='Отчество'),
        ),
        migrations.AlterField(
            model_name='student',
            name='entery_score',
            field=models.IntegerField(verbose_name='Средний балл'),
        ),
    ]
