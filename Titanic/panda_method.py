import pandas

pd = pandas.read_csv("/Users/xyan/pycharm_project/Kaggle_learning/Titanic/data/train.csv")
pd = pd.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)
print(pd.Sex[0])
pd.to_csv("test.csv")
