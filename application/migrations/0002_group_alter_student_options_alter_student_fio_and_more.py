# Generated by Django 5.1.2 on 2024-10-13 07:45

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Group',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(verbose_name='Название')),
                ('form_of_education', models.TextField(verbose_name='Форма обучения')),
            ],
        ),
        migrations.AlterModelOptions(
            name='student',
            options={'verbose_name': 'Студент', 'verbose_name_plural': 'Студенты'},
        ),
        migrations.AlterField(
            model_name='student',
            name='fio',
            field=models.TextField(verbose_name='ФИО'),
        ),
        migrations.AlterField(
            model_name='student',
            name='group_name',
            field=models.TextField(verbose_name='Группа'),
        ),
        migrations.AlterField(
            model_name='student',
            name='record_number',
            field=models.TextField(verbose_name='Номер зачетки'),
        ),
        migrations.AddField(
            model_name='student',
            name='group',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.group'),
        ),
    ]