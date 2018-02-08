import csv

import pandas as pd
import math
import random


def remove_useless_data_create_new_train_data(input_file, output_file):
    pd_reader = pd.read_csv(input_file)
    pd_reader = pd_reader.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', ], 1)

    Sex_list = pd_reader.Sex.values.tolist()
    Sex_male_list_number, Sex_female_list_number = [], []
    # convert the sex string to one-hot encoding
    for item in Sex_list:
        if item == 'male':
            Sex_male_list_number.append(1), Sex_female_list_number.append(0)
        else:
            Sex_male_list_number.append(0), Sex_female_list_number.append(1)
    pd_reader['MaleSex'] = pd.DataFrame(Sex_male_list_number)
    pd_reader['FemaleSex'] = pd.DataFrame(Sex_female_list_number)
    pd_reader = pd_reader.drop(['Sex'], 1)

    # convert the Embarked string to one-hot encoding
    Embarked_list = pd_reader.Embarked.values.tolist()
    Embarked_S_list_number, Embarked_C_list_number, Embarked_Q_list_number = [], [], []
    for item in Embarked_list:
        if item == 'S':
            Embarked_S_list_number.append(1), Embarked_C_list_number.append(0), Embarked_Q_list_number.append(0)
        elif item == 'C':
            Embarked_C_list_number.append(1), Embarked_S_list_number.append(0), Embarked_Q_list_number.append(0)
        elif item == 'Q':
            Embarked_Q_list_number.append(1), Embarked_C_list_number.append(0), Embarked_S_list_number.append(0)
    pd_reader['Embarked_S'], pd_reader['Embarked_C'], pd_reader['Embarked_Q'] = pd.DataFrame(
        Embarked_S_list_number), pd.DataFrame(Embarked_C_list_number), pd.DataFrame(Embarked_Q_list_number)
    pd_reader = pd_reader.drop(['Embarked'], 1)

    # fill the age with random value
    age_list = pd_reader.Age.values.tolist()
    # print(age_list)
    age_list_new = []
    for age in age_list:
        if math.isnan(age):
            age_list_new.append(random.randint(15, 55))
        else:
            age_list_new.append(int(age))
    pd_reader.Age = pd.DataFrame(age_list_new)

    # convert pclass string to one-hot encoding
    pclass_list = pd_reader.Pclass.values.tolist()
    pclass_one_list, pclass_two_list, pclass_three_list = [], [], []
    for item in pclass_list:
        if item in pclass_list:
            if item == 1:
                pclass_one_list.append(1), pclass_two_list.append(0), pclass_three_list.append(0)
            elif item == 2:
                pclass_one_list.append(0), pclass_two_list.append(1), pclass_three_list.append(0)
            elif item == 3:
                pclass_one_list.append(0), pclass_two_list.append(0), pclass_three_list.append(1)
    pd_reader['pclass_one'], pd_reader['pclass_two'], pd_reader['pclass_three'] = pd.DataFrame(
        pclass_one_list), pd.DataFrame(pclass_two_list), pd.DataFrame(pclass_three_list)
    pd_reader = pd_reader.drop(['Pclass'], 1)

    print(pd_reader)
    pd_reader.to_csv(output_file, index=False)


# def remove_useless_data_create_new_test_data(input_file, output_file):
#     pd_reader = pd.read_csv(input_file)
#     pd_reader = pd_reader.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', ], 1)
#
#     Sex_list = pd_reader.Sex.values.tolist()
#     Sex_male_list_number, Sex_female_list_number = [], []
#     # convert the sex string to one-hot encoding
#     for item in Sex_list:
#         if item == 'male':
#             Sex_male_list_number.append(1), Sex_female_list_number.append(0)
#         else:
#             Sex_male_list_number.append(0), Sex_female_list_number.append(1)
#     pd_reader['MaleSex'] = pd.DataFrame(Sex_male_list_number)
#     pd_reader['FemaleSex'] = pd.DataFrame(Sex_female_list_number)
#     pd_reader = pd_reader.drop(['Sex'], 1)
#
#     Embarked_list = pd_reader.Embarked.values.tolist()
#     Embarked_S_list_number, Embarked_C_list_number, Embarked_Q_list_number = [], [], []
#     for item in Embarked_list:
#         if item == 'S':
#             Embarked_S_list_number.append(1), Embarked_C_list_number.append(0), Embarked_Q_list_number.append(0)
#         elif item == 'C':
#             Embarked_C_list_number.append(1), Embarked_S_list_number.append(0), Embarked_Q_list_number.append(0)
#         elif item == 'Q':
#             Embarked_Q_list_number.append(1), Embarked_C_list_number.append(0), Embarked_S_list_number.append(0)
#     pd_reader['Embarked_S'], pd_reader['Embarked_C'], pd_reader['Embarked_Q'] = pd.DataFrame(
#         Embarked_S_list_number), pd.DataFrame(Embarked_C_list_number), pd.DataFrame(Embarked_Q_list_number)
#     pd_reader = pd_reader.drop(['Embarked'], 1)
#
#     # fill the age with random value
#     age_list = pd_reader.Age.values.tolist()
#     # print(age_list)
#     age_list_new = []
#     for age in age_list:
#         if math.isnan(age):
#             age_list_new.append(random.randint(15, 55))
#         else:
#             age_list_new.append(int(age))
#
#     pd_reader.Age = pd.DataFrame(age_list_new)
#     print(pd_reader)
#     pd_reader.to_csv(output_file, index=False)


def split_scv_file_with_age():
    with open("./data/train_new.csv", "r", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        content = [line for line in csv_reader]
    age_content = [line for line in content if line[3]]
    without_age_content = [line for line in content if not line[3]]
    with open("./data/train_age.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in age_content:
            line[2] = 1 if line[2] == "male" else 0
            csv_writer.writerow(line)
    with open("train_without_age.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in without_age_content:
            csv_writer.writerow(line)

if __name__ == '__main__':
    input_file, output_file = "./data/train.csv", "./data/train_new.csv"
    remove_useless_data_create_new_train_data(input_file, output_file)
    input_file, output_file = "./data/test.csv", "./data/test_new.csv"
    remove_useless_data_create_new_train_data(input_file, output_file)
    # split_scv_file_with_age()
