CREATE EXTERNAL TABLE employees (
    employeeid INT,
    managerid INT,
    firstname STRING,
    lastname STRING,
    fullname STRING,
    jobtitle STRING,
    organizationlevel INT,
    maritalstatus STRING,
    gender STRING,
    territory STRING,
    country STRING,
    `group` STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://your-bucket/employees/';