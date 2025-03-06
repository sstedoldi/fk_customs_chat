CREATE EXTERNAL TABLE orders (
    salesorderid INT,
    salesorderdetailid INT,
    orderdate STRING,
    duedate STRING,
    shipdate STRING,
    employeeid INT,
    customerid INT,
    subtotal DOUBLE,
    taxamt DOUBLE,
    freight DOUBLE,
    totaldue DOUBLE,
    productid INT,
    orderqty INT,
    unitprice DOUBLE,
    unitpricediscount DOUBLE,
    linetotal DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://your-bucket/orders/'; 