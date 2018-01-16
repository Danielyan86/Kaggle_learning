import pandas

pd_csv = pandas.read_csv("./data/train.csv")
# print(pd_csv)
pd_csv = pd_csv.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Fare', 'Embarked', ], 1)
pd_csv_without_age = (pd_csv[pd_csv.isnull().any(axis=1)])
pd_csv_without_age = pd_csv_without_age.drop(['Age'], 1)
pd_csv_without_age = pd_csv_without_age.drop(pd_csv_without_age.columns[0], axis=1)
print(pd_csv_without_age)

pd_csv_with_age = pd_csv.dropna(axis=0, how='any')
print(pd_csv_with_age)

pd_csv_with_age.to_csv("./data/test_with_age.csv", index=False)
pd_csv_without_age.to_csv("./data/test_without_age.csv", index=False)
