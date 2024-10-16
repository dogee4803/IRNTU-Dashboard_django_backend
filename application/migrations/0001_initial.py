# Generated by Django 5.1.2 on 2024-10-13 13:46

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Speciality',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.TextField(max_length=30, verbose_name='Код')),
                ('name', models.TextField(max_length=30, verbose_name='Название специальности')),
            ],
            options={
                'verbose_name': 'Специальность',
                'verbose_name_plural': 'Специальности',
            },
        ),
        migrations.CreateModel(
            name='Subject',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.TextField(max_length=30, verbose_name='Код')),
                ('name', models.TextField(max_length=30, verbose_name='Название предмета')),
            ],
            options={
                'verbose_name': 'Предмет',
                'verbose_name_plural': 'Предметы',
            },
        ),
        migrations.CreateModel(
            name='Teacher',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_name', models.TextField(max_length=30, verbose_name='Фамилия')),
                ('first_name', models.TextField(max_length=30, verbose_name='Имя')),
                ('patronymic', models.TextField(max_length=30, verbose_name='Отчество')),
                ('service_number', models.TextField(max_length=15, verbose_name='Служебный номер')),
            ],
            options={
                'verbose_name': 'Преподаватель',
                'verbose_name_plural': 'Преподаватели',
            },
        ),
        migrations.CreateModel(
            name='Group',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(max_length=30, verbose_name='Название группы')),
                ('group_number', models.IntegerField(verbose_name='Номер группы')),
                ('form_of_education', models.TextField(max_length=15, verbose_name='Форма обучения')),
                ('speciality', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='application.speciality')),
            ],
            options={
                'verbose_name': 'Группа',
                'verbose_name_plural': 'Группы',
            },
        ),
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_name', models.TextField(max_length=30, verbose_name='Фамилия')),
                ('first_name', models.TextField(max_length=30, verbose_name='Имя')),
                ('patronymic', models.TextField(max_length=30, verbose_name='Отчество')),
                ('record_number', models.TextField(max_length=8, verbose_name='Номер зачетки')),
                ('group', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.group')),
            ],
            options={
                'verbose_name': 'Студент',
                'verbose_name_plural': 'Студенты',
            },
        ),
        migrations.CreateModel(
            name='Student_Subject',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('grade', models.IntegerField(verbose_name='Оценка')),
                ('student', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.student')),
                ('subject', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.subject')),
            ],
            options={
                'verbose_name': 'Оценка студента',
                'verbose_name_plural': 'Оценки студентов',
            },
        ),
        migrations.AddField(
            model_name='subject',
            name='teacher',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.teacher'),
        ),
        migrations.CreateModel(
            name='Teacher_Subject',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('subject', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.subject')),
                ('teacher', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='application.teacher')),
            ],
            options={
                'verbose_name': 'Предмет преподавателя',
                'verbose_name_plural': 'Предметы преподавателей',
            },
        ),
    ]
