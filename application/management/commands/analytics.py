from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.mlab as mlab
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from django.db import connection

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Run analytics script'
    def get_students_from_db(self):
        query = """
            WITH SemesterGrades AS (
                SELECT 
                    gd.Student_ID,
                    hps.Semester,
                    AVG(CAST(gd.Grade AS DECIMAL(10,2))) AS Avg_Grade
                FROM Grades gd
                JOIN Form_control fc ON gd.FC_ID = fc.FC_ID
                JOIN Hours_per_semest hps ON fc.HPS_ID = hps.HPS_ID
                WHERE 
                    fc.Form != 'Зачет' 
                    AND TRY_CAST(gd.Grade AS DECIMAL(10,2)) IS NOT NULL
                GROUP BY gd.Student_ID, hps.Semester
            )

            SELECT 
                sp.Title AS Speciality,
                g.Title AS Group_Name,
                s.Student_ID,
                s.Name,
                DATEDIFF(YEAR, s.Birth_date, GETDATE()) AS Age,
                CASE WHEN a.Student_ID IS NOT NULL THEN 1 ELSE 0 END AS Is_Academic, 
                s.Middle_value_of_sertificate, 
                s.Entry_score, 
                COALESCE(r.Score, 0) AS Rating_score,
                dpl.Grade AS Diploma_grade,

                -- Оценки по семестрам
                COALESCE(MAX(CASE WHEN sg.Semester = 1 THEN sg.Avg_Grade END), 0) AS Semester_1_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 2 THEN sg.Avg_Grade END), 0) AS Semester_2_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 3 THEN sg.Avg_Grade END), 0) AS Semester_3_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 4 THEN sg.Avg_Grade END), 0) AS Semester_4_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 5 THEN sg.Avg_Grade END), 0) AS Semester_5_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 6 THEN sg.Avg_Grade END), 0) AS Semester_6_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 7 THEN sg.Avg_Grade END), 0) AS Semester_7_Grade,
                COALESCE(MAX(CASE WHEN sg.Semester = 8 THEN sg.Avg_Grade END), 0) AS Semester_8_Grade,

                -- Список долгов
                COALESCE(( 
                    SELECT DISTINCT dsc.Disciple_name + ', ' 
                    FROM Debts d 
                    JOIN Hours_per_semest hps ON d.HPS_ID = hps.HPS_ID 
                    JOIN Disciples dsc ON hps.Disciple_ID = dsc.Disciple_ID 
                    WHERE d.Student_ID = s.Student_ID 
                    FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 'Нет долгов') AS Debts_List,

                -- Средняя оценка за практику
                COALESCE(( 
                    SELECT AVG(CAST(p.Grade AS DECIMAL(10,2))) 
                    FROM Practise p 
                    WHERE p.Student_ID = s.Student_ID 
                ), 0) AS Avg_Practise_Grade,

                -- Средняя посещаемость
                COALESCE(( 
                    SELECT AVG(att.Percent_of_attendance) 
                    FROM Attendance att 
                    WHERE att.Student_ID = s.Student_ID 
                ), 0) AS Avg_Attendance,

                -- Все оценки по предметам
                COALESCE((
                    SELECT 
                        STUFF((
                            SELECT '; ' + dsc.Disciple_name + ': ' + CAST(gd.Grade AS NVARCHAR)
                            FROM Grades gd
                            JOIN Form_control fc ON gd.FC_ID = fc.FC_ID
                            JOIN Hours_per_semest hps ON fc.HPS_ID = hps.HPS_ID
                            JOIN Disciples dsc ON hps.Disciple_ID = dsc.Disciple_ID
                            WHERE gd.Student_ID = s.Student_ID
                            AND fc.Form != 'Зачет'
                            AND TRY_CAST(gd.Grade AS DECIMAL(10,2)) IS NOT NULL
                            FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 2, '')
                ), 'Нет оценок') AS All_Grades_Per_Subject

                FROM Student s
                JOIN [Group] g ON s.Group_ID = g.Group_ID
                JOIN Education_plan ep ON g.Plan_ID = ep.Plan_ID
                JOIN Specialty sp ON ep.Code = sp.Code
                LEFT JOIN Academ a ON s.Student_ID = a.Student_ID
                LEFT JOIN Rating r ON s.Rating_ID = r.Rating_ID  
                LEFT JOIN Diploma dpl ON s.Student_ID = dpl.Student_ID
                LEFT JOIN SemesterGrades sg ON s.Student_ID = sg.Student_ID

                GROUP BY 
                    sp.Title, g.Title, s.Student_ID, s.Name, s.Birth_date, 
                    s.Middle_value_of_sertificate, s.Entry_score, r.Score, 
                    dpl.Grade, a.Student_ID

                ORDER BY s.Student_ID;
            """
            
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            
            return pd.DataFrame(data, columns=columns)

    def handle(self, *args, **options):
        

        students = self.get_students_from_db()

        # Функция для парсинга строки в словарь
        def parse_subject_grades(grade_str):
            if pd.isna(grade_str):
                return {}
            parts = grade_str.split(';')
            grades = {}
            for part in parts:
                if ':' in part:
                    subject, grade = part.strip().split(':')
                    grades[subject.strip()] = float(grade.strip())
            return grades

        # Применим ко всем строкам
        students['Grades_Dict'] = students['All_Grades_Per_Subject'].apply(parse_subject_grades)


        def mean_grade(grades_dict):
            if not grades_dict:
                return None
            return sum(grades_dict.values()) / len(grades_dict)

        print(students['Grades_Dict'])
        students['Mean_Subject_Grade'] = students['Grades_Dict'].apply(mean_grade)
        print('Работа с долгами')
        debts_list=[]
        for i in range(1325):
            debts_list.append(students['Debts_List'][i].split(','))
            print(debts_list)

        for i in range(len(students)):
            if isinstance(students['Grades_Dict'][i], dict) and isinstance(debts_list[i], list):
                for subject in debts_list[i]:
                    subject = subject.strip()  # удаляем лишние пробелы
                    if subject in students['Grades_Dict'][i]:
                        students['Grades_Dict'][i][subject] = 'Долг'
        print(students['Grades_Dict'][1])
        # Добавляем отдельные столбцы для предметов и оценок
        all_subjects = set()  # Используем множество для уникальных предметов
        for grades_dict in students['Grades_Dict']:
            all_subjects.update(grades_dict.keys())

        for subject in all_subjects:
            students[subject] = None

        for index, row in students.iterrows():
            grades_dict = row['Grades_Dict']
            for subject, grade in grades_dict.items():
                students.loc[index, subject] = grade

        # Заменяем NaN и None на 0 во всей таблице
        students = students.fillna(0)

        # Выводим всю таблицу
        print(students)

        # Или сохранение в CSV (рекомендуется для больших таблиц):
        students.to_csv('students_with_all_columns.csv', index=False, sep=';', encoding='utf-8')

        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import pairwise_distances
        import matplotlib.pyplot as plt

        # 1. Подготовка данных для кластеризации (3 курс)
        semester_5_6_data = students.iloc[364:524].copy()  # Копируем данные студентов 3 курса

        # 2. Выбор признаков для кластеризации (только предметы 3 курса)
        all_subjects_3rd_year = set()
        for grades_dict in semester_5_6_data['Grades_Dict']:  # Итерируемся только по данным 3 курса
            all_subjects_3rd_year.update(grades_dict.keys())  # Добавляем все предметы 3 курса в множество

        features_for_clustering = list(all_subjects_3rd_year)  # Преобразуем множество в список

        # 3. Замена "Долг" на -1 (без изменений)
        for subject in features_for_clustering:  # Итерируемся по предметам 3 курса
            semester_5_6_data[subject] = semester_5_6_data[subject].replace('Долг', -1).astype(float) 

        # 4. Обработка нулей (замена на NaN)
        for subject in features_for_clustering:  # Итерируемся по предметам 3 курса
            semester_5_6_data[subject] = semester_5_6_data[subject].replace(0, np.nan) 


        # 5. Масштабирование данных (с использованием маски для исключения NaN)
        numeric_data = semester_5_6_data[features_for_clustering].copy()  # Копируем данные для масштабирования

        # Создание маски (исключает NaN)
        mask = ~numeric_data.isna()  # True, если значение не NaN

        scaler = StandardScaler()  # Создаем объект StandardScaler
        scaled_values = scaler.fit_transform(numeric_data[mask])  # Масштабируем данные, используя маску
        numeric_data[mask] = scaled_values  # Возвращаем масштабированные данные в DataFrame
        scaled_data = numeric_data  # Переименовываем для ясности

        # 6. Определение оптимального числа кластеров (метод локтя)
        inertia = []  # Список для хранения значений инерции
        for k in range(1, 11):  # Проверяем k от 1 до 10
            kmeans = KMeans(n_clusters=k, random_state=42)  # Создаем объект KMeans
            kmeans.fit(scaled_data.fillna(0))  # Заполняем NaN нулями только для метода локтя и обучаем модель
            inertia.append(kmeans.inertia_)  # Добавляем значение инерции в список

        plt.plot(range(1, 11), inertia, marker='o')  # Строим график
        plt.xlabel("Количество кластеров (k)")
        plt.ylabel("Инерция")
        plt.title("Метод локтя для кластеризации предметов")
        plt.show()  # Показываем график


        k = 2  # Выбираем k на основе графика (хотя там вообще один кластер можно бы поставить)

        # 7. Применение KMeans для кластеризации предметов (с учетом NaN и транспонированием)
        def nan_euclidean_distance(X, Y):  # Функция для вычисления расстояния с учетом NaN
            return np.sqrt(np.nansum((X - Y)**2, axis=0))  # axis=0 для кластеризации предметов

        scaled_data_transposed = scaled_data.T  # Транспонируем данные для кластеризации предметов

        kmeans = KMeans(n_clusters=k, random_state=42)  # Создаем объект KMeans
        distance_matrix = pairwise_distances(scaled_data_transposed.fillna(0), metric=nan_euclidean_distance)  # Вычисляем матрицу расстояний, заполняя NaN нулями только здесь

        kmeans.fit(distance_matrix)  # Обучаем KMeans на матрице расстояний
        subject_clusters = kmeans.labels_  # Получаем метки кластеров для предметов

        # 8. Добавление информации о кластерах для предметов
        subject_cluster_mapping = dict(zip(all_subjects_3rd_year, subject_clusters)) # Создаем словарь {предмет: номер_кластера}
        print("\nСоответствие предметов кластерам:")
        for subject, cluster in subject_cluster_mapping.items(): # Выводим результаты
            print(f"{subject}: Кластер {cluster}")

        # 1. Подготовка данных (3 курс)
        semester_5_6_data = students.iloc[364:728].copy()

        # 2. Выбор признаков (только предметы 3 курса)
        all_subjects_3rd_year = set()
        for grades_dict in semester_5_6_data['Grades_Dict']:
            all_subjects_3rd_year.update(grades_dict.keys())
        features_for_clustering = list(all_subjects_3rd_year)


        # 3. Замена "Долг" на -1
        for subject in features_for_clustering:
            semester_5_6_data[subject] = semester_5_6_data[subject].replace('Долг', -1).astype(float)

        # 5. Масштабирование данных 
        numeric_data = semester_5_6_data[features_for_clustering].copy()  # Копируем данные для масштабирования

        # Создание маски (исключает NaN)
        mask = ~numeric_data.isna()  # True, если значение не NaN

        scaler = StandardScaler()  # Создаем объект StandardScaler
        scaled_values = scaler.fit_transform(numeric_data[mask])  # Масштабируем данные, используя маску
        numeric_data[mask] = scaled_values  # Возвращаем масштабированные данные в DataFrame
        scaled_data = numeric_data  # Переименовываем для ясности

        import pandas as pd
        import numpy as np

        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report

        # --- Подготовка данных ---

        # 1. Данные 3 курса (для обучения модели)
        train_data = students.iloc[364:524].copy()  # 3 курс - обучающая выборка
        train_features = list(all_subjects_3rd_year)  # только предметы 3 курса

        # 2. Данные 2 курса (для предсказания)
        predict_data = students.iloc[525:760].copy()  # 2 курс - данные для предсказания
        predict_features = list(set(all_subjects_3rd_year).intersection(predict_data.columns))  # Пересечение предметов 3 и 2 курсов

        # 3. Замена "Долг" на -1 (в обоих датасетах)
        for subject in all_subjects_3rd_year: 
            if subject in train_data.columns:
                train_data[subject] = train_data[subject].replace('Долг', -1).astype(float)
            if subject in predict_data.columns:
                predict_data[subject] = predict_data[subject].replace('Долг', -1).astype(float)

        # 4. Обработка нулей (замена на NaN) (в обоих датасетах)
        for subject in all_subjects_3rd_year:
            if subject in train_data.columns:
                train_data[subject] = train_data[subject].replace(0, np.nan)
            if subject in predict_data.columns:
                predict_data[subject] = predict_data[subject].replace(0, np.nan)

        # --- Обучение и предсказание ---
        predictions = {}
        for subject in predict_features:  # Итерируемся по предметам, общим для 2 и 3 курсов

            # 6. Подготовка данных для модели (преобразование оценок в категории)
            X_train = train_data[train_features].copy()  # Все предметы 3 курса как признаки.
            y_train = train_data[subject].apply(lambda x: 'Высокая' if x >= 4 else ('Низкая' if x < 4 and x > -1 else 'Долг'))
            X_predict = predict_data[predict_features].copy()  # Все общие предметы 2 курса как признаки

            # 7. One-Hot Encoding (для категориальных признаков, если есть)
            X_train = pd.get_dummies(X_train)
            X_predict = pd.get_dummies(X_predict)
            # ensure same columns in training and prediction
            X_predict = X_predict.reindex(columns=X_train.columns, fill_value=0)

            # 8. Обучение модели
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train.fillna(0), y_train)  # Обучаем модель, заполняя NaN нулями

            # 9. Предсказание
            y_pred = model.predict(X_predict.fillna(0))  # Предсказываем, заполняя NaN нулями
            predictions[subject] = y_pred

            print(f"Classification report for {subject}:")
            print(classification_report(predict_data[subject].apply(lambda x: 'Высокая' if x >= 4 else ('Низкая' if x < 4 and x > -1 else 'Долг')), y_pred))

        # 10. Добавление предсказаний в DataFrame
        for subject, pred in predictions.items():
            predict_data[f'{subject}_prediction'] = pred

        # 11. Визуализация результатов
        num_subjects = len(predict_features)  # Получаем количество предметов
        cols = 3  # Количество столбцов для подграфиков
        rows = (num_subjects + cols - 1) // cols  # Вычисляем количество строк

        plt.figure(figsize=(16, 5 * rows))  # Увеличиваем высоту фигуры в зависимости от количества строк
        for i, subject in enumerate(predict_features, 1):
            plt.subplot(rows, cols, i)  # Настройка сетки подграфиков
            sns.countplot(data=predict_data, x=f'{subject}_prediction', palette='viridis')
            plt.title(f'Предсказания для {subject}')
            plt.xlabel('Категория')
            plt.ylabel('Количество')
        plt.tight_layout()
        plt.show()


        # 12. Вывод результатов
        print(predict_data)







        # Выбираем нужные столбцы
        columns_to_save = [
            'Speciality', 'Group_Name', 'Student_ID', 'Name', 'Age', 'Is_Academic',
            'Middle_value_of_sertificate', 'Entry_score', 'Rating_score', 'Diploma_grade'
        ] + [col for col in predict_data.columns if '_prediction' in col]

        # Создаем DataFrame с нужными столбцами
        result_data = predict_data[columns_to_save]

        # Удаляем "_prediction" из названий столбцов
        result_data.columns = [
            col.replace("_prediction", "") if "_prediction" in col else col
            for col in result_data.columns
        ]

        # Сохраняем в CSV
        result_data.to_csv('predictions_results.csv', index=False, sep=';', encoding='utf-8')
        

