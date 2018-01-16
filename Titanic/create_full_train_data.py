import csv

import pandas as pd


def remove_useless_data_create_new_train_data(input_file, output_file):
    # remove name
    # pattern = re.compile(r'(\".*\")')
    # with open(input_file) as f:
    #     content = [pattern.sub("", n) for n in f.readlines()][1:]
    # print(content)
    # for item in content:
    #     item_list = item.split(",")
    #     if len(item_list) != 11:
    #         raise AssertionError("failed")
    # content = [line.split(",")[1:9] for line in content]  # remove the first and last one
    #
    # # remove useless data and write to csv file
    # with open(output_file, "w", newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for line in content:
    #         line.pop(1)
    #         line.pop(-2)
    #         line[1] = 1 if line[1] == "male" else 0
    #         if not line[2]:
    #             line[2] = random.randint(20, 50)
    #         csv_writer.writerow(line)
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
            Embarked_list_number.append(0)
        elif item == 'C':
            Embarked_list_number.append(1)
        elif item == 'Q':
            Embarked_list_number.append(2)
        else:
            Embarked_list_number.append(2)
    pd_reader.Embarked = pd.DataFrame(Embarked_list_number)
    print(pd_reader)


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
    # split_scv_file_with_age()
