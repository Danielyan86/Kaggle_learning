import csv
import random
import re


def remove_useless_data_create_new_test_data(input_file, output_file):
    # remove name
    pattern = re.compile(r'(\".*\")')
    with open(input_file) as f:
        content = [pattern.sub("", n) for n in f.readlines()][1:]
    print(content)
    for item in content:
        item_list = item.split(",")
        if len(item_list) != 11:
            raise AssertionError("failed")
    content = [line.split(",")[1:9] for line in content]  # remove the first and last one

    # remove useless data and write to csv file
    with open(output_file, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in content:
            line.pop(1)
            line.pop(-2)
            line[1] = 1 if line[1] == "male" else 0
            if not line[2]:
                line[2] = random.randint(20, 50)
            csv_writer.writerow(line)


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
    input_file, output_file = "./data/test.csv", "./data/test_new.csv"
    remove_useless_data_create_new_test_data(input_file, output_file)
    # split_scv_file_with_age()
