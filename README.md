### Diagnostics duration train and prediction using catboost regressor and return to 1c

## install:

git clone https://github.com/format37/1c_ml_regression_diagnostics.git
sudo chmod -R 777 cgi-bin/

## run server:

sh start.sh

## request to server:
```
# replace ip to yours
http://10.2.4.87:8000/cgi-bin/test.py
```

## issues:

#/usr/bin/env: ‘python3\r’: No such file or directory
https://stackoverflow.com/questions/1523427/what-is-the-common-header-format-of-python-files
solution:
```
rm cgi-bin/test.py

nano cgi-bin/test.py

#!/usr/bin/env python3
print("Content-type: text/html")
print()
print("Hello")

#^xy

sudo chmod -R 777 cgi-bin/
```
