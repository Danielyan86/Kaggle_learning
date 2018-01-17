import csv

import pandas as pd
import math
import random


def remove_useless_data_create_new_train_data(input_file, output_file):
    pd_reader = pd.read_csv(input_file)
    pd_reader = pd_reader.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', ], 1)
    print(pd_reader)
    print(pd_reader.Sex)
    Sex_list = pd_reader.Sex.values.tolist()
    Sex_list_number = []
    for item in Sex_list:
        if item == 'male':
            Sex_list_number.append(0)
        else:
            Sex_list_number.append(1)
    pd_reader.Sex = pd.DataFrame(Sex_list_number)

    Embarked_list = pd_reader.Embarked.values.tolist()
    Embarked_list_number = []
    for item in Embarked_list:
        if item == 'S':
            Embarked_list_number.append(1)
        elif item == 'C':
            Embarked_list_number.append(2)
        elif item == 'Q':
            Embarked_list_number.append(3)
        else:
            Embarked_list_number.append(3)
    pd_reader.Embarked = pd.DataFrame(Embarked_list_number)

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
    print(pd_reader)
    pd_reader.to_csv(output_file, index=False)


def remove_useless_data_create_new_test_data(input_file, output_file):
    pd_reader = pd.read_csv(input_file)
    pd_reader = pd_reader.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', ], 1)
    print(pd_reader)
    print(pd_reader.Sex)
    Sex_list = pd_reader.Sex.values.tolist()
    Sex_list_number = []
    for item in Sex_list:
        if item == 'male':
            Sex_list_number.append(0)
        else:
            Sex_list_number.append(1)
    pd_reader.Sex = pd.DataFrame(Sex_list_number)

    Embarked_list = pd_reader.Embarked.values.tolist()
    Embarked_list_number = []
    for item in Embarked_list:
        if item == 'S':
            Embarked_list_number.append(1)
        elif item == 'C':
            Embarked_list_number.append(2)
        elif item == 'Q':
            Embarked_list_number.append(3)
        else:
            Embarked_list_number.append(3)
    pd_reader.Embarked = pd.DataFrame(Embarked_list_number)

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
    print(pd_reader)
    pd_reader.to_csv(output_file, index=False)


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

    # print(number, len(number))


if __name__ == '__main__':
    input_file, output_file = "./data/train.csv", "./data/train_new.csv"
    remove_useless_data_create_new_train_data(input_file, output_file)
    input_file, output_file = "./data/test.csv", "./data/test_new.csv"
    remove_useless_data_create_new_test_data(input_file, output_file)
    # split_scv_file_with_age()
