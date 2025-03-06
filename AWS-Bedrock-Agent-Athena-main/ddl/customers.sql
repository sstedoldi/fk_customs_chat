CREATE EXTERNAL TABLE customers (
    customerid INT,
    firstname STRING,
    lastname STRING,
    fullname STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://your-bucket/customers/'
TBLPROPERTIES ('skip.header.line.count'='0'); 