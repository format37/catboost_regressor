### Catboost regressor model train, predict and call python from 1c

## mysql db prepare
```
sudo mysql -u root
CREATE USER '1c'@'localhost' IDENTIFIED BY 'PASS';
GRANT ALL PRIVILEGES ON * . * TO '1c'@'localhost';
FLUSH PRIVILEGES;
exit
sudo mysql -u 1c -p
CREATE DATABASE ml;
CREATE TABLE regression( id INT NOT NULL AUTO_INCREMENT, row INT NOT NULL, field INT NOT NULL, value VARCHAR(512), PRIMARY KEY (id) );
alter table regression convert to character set utf8mb4 collate utf8mb4_unicode_ci;
ALTER TABLE regression MODIFY value LONGTEXT;
```

## requirements:
```
sudo su
pip3 install pandas --user
pip3 install Cython --user
pip3 install catboost --user
```

# installation
```
git clone https://github.com/format37/1c_ml_regression_diagnostics.git
sudo chmod -R 777 cgi-bin/
```

## run server:
```
sh start.sh
```

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

1c should be a 64-bit app
examples to setup mysql^
```
create table regression ( id int not null auto_increment, request varchar(36), row int(11), field int(4), value varchar(36), primary key (id) )auto_increment=100;

sudo nano /etc/mysql/my.cnf
sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
comment bind-address = 127.0.0.1 using the # symbol

CREATE USER '1c'@'%' IDENTIFIED BY 'PASS';
GRANT ALL PRIVILEGES ON *.* TO '1c'@'%';
FLUSH PRIVILEGES;

mysql-connector-odbc-5.3.4-winx64.msi
```
https://ru.fileerrors.com/delete-cr-lf-from-text-file.html
