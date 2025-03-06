import json
import boto3
import time
from typing import Dict, List, Optional

athena = boto3.client('athena')
s3 = boto3.client('s3')

DATABASE = 'default'
WORKGROUP = 'bedrock_demo'  # Add workgroup name

def execute_query(query: str) -> List[Dict]:
    """Execute Athena query and return results"""
    query_execution = athena.start_query_execution(
        QueryString=query,
        WorkGroup=WORKGROUP,  # Specify workgroup
        QueryExecutionContext={
            'Database': DATABASE
        }
    )
    
    execution_id = query_execution['QueryExecutionId']
    
    # Wait for query to complete
    while True:
        response = athena.get_query_execution(QueryExecutionId=execution_id)
        state = response['QueryExecution']['Status']['State']
        
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
            
        time.sleep(1)
    
    if state != 'SUCCEEDED':
        print(response)
        raise Exception(f"Query failed with state: {state}")
    
    # Get results
    results = athena.get_query_results(QueryExecutionId=execution_id)
    
    # Process results
    columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
    rows = []
    
    for row in results['ResultSet']['Rows'][1:]:  # Skip header row
        data = {}
        for i, value in enumerate(row['Data']):
            data[columns[i]] = value.get('VarCharValue', '')
        rows.append(data)
    
    return rows

def get_customer_by_name(name: str) -> Optional[Dict]:
    """Search for customer by name using Athena"""
    query = f"""
    SELECT customerid, firstname, lastname, fullname 
    FROM customers 
    WHERE LOWER(fullname) LIKE LOWER('%{name}%')
    OR LOWER(firstname) LIKE LOWER('%{name}%')
    OR LOWER(lastname) LIKE LOWER('%{name}%')
    LIMIT 1
    """
    
    results = execute_query(query)
    return results[0] if results else None

def get_customer_orders(customer_id: str) -> List[Dict]:
    """Get all orders for a customer using Athena"""
    query = f"""
    SELECT 
        salesorderid,
        salesorderdetailid,
        orderdate,
        duedate,
        shipdate,
        employeeid,
        customerid,
        CAST(subtotal AS DOUBLE) as subtotal,
        CAST(taxamt AS DOUBLE) as taxamt,
        CAST(freight AS DOUBLE) as freight,
        CAST(totaldue AS DOUBLE) as totaldue,
        productid,
        CAST(orderqty AS INTEGER) as orderqty,
        CAST(unitprice AS DOUBLE) as unitprice,
        CAST(unitpricediscount AS DOUBLE) as unitpricediscount,
        CAST(linetotal AS DOUBLE) as linetotal
    FROM orders 
    WHERE customerid = CAST('{customer_id}' as INT) limit 20
    """
    
    return execute_query(query)

def get_all_customers_and_orders() -> List[Dict]:
    """Get all customers and their orders using Athena"""
    query = """
    SELECT DISTINCT
        c.customerid,
        c.firstname,
        c.lastname,
        c.fullname
    FROM customers c
    """
    
    customers = execute_query(query)
    result = []
    
    for customer in customers:
        orders = get_customer_orders(customer['customer_id'])
        result.append({
            'customer': customer,
            'orders': orders
        })
    
    return result

def lambda_handler(event, context):
    """Main Lambda handler for API Gateway proxy integration and Bedrock agent"""
    try:
        # Check if this is a Bedrock agent request
        if 'messageVersion' in event:
            # Handle Bedrock agent request
            api_path = event.get('apiPath')
            if api_path == '/orders/customer/{name}':
                # Extract name parameter
                name_param = next((p for p in event.get('parameters', []) if p['name'] == 'name'), None)
                if not name_param:
                    return {
                        'messageVersion': '1.0',
                        'response': {
                            'actionGroup': event['actionGroup'],
                            'apiPath': api_path,
                            'httpMethod': 'GET',
                            'httpStatusCode': 400,
                            'responseBody': {
                                'application/json': {
                                    'body': json.dumps({'error': 'Name parameter is required'})
                                }
                            }
                        }
                    }

                name = name_param['value']
                customer = get_customer_by_name(name)
                
                if not customer:
                    return {
                        'messageVersion': '1.0',
                        'response': {
                            'actionGroup': event['actionGroup'],
                            'apiPath': api_path,
                            'httpMethod': 'GET',
                            'httpStatusCode': 404,
                            'responseBody': {
                                'application/json': {
                                    'body': json.dumps({'error': 'Customer not found'})
                                }
                            }
                        }
                    }
                
                orders = get_customer_orders(customer['customerid'])
                return {
                    'messageVersion': '1.0',
                    'response': {
                        'actionGroup': event['actionGroup'],
                        'apiPath': api_path,
                        'httpMethod': 'GET',
                        'httpStatusCode': 200,
                        'responseBody': {
                            'application/json': {
                                'body': json.dumps({
                                    'customer': customer,
                                    'orders': orders
                                })
                            }
                        }
                    },
                    'sessionAttributes': event.get('sessionAttributes', {}),
                    'promptSessionAttributes': event.get('promptSessionAttributes', {})
                }

        # Handle API Gateway request
        path_parameters = event.get('pathParameters', {}) or {}
        resource_path = event.get('resource', '')
        http_method = event.get('httpMethod', '')

        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'GET,OPTIONS'
        }

        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': ''
            }

        if http_method == 'GET':
            if resource_path == '/orders/customer/{name}':
                name = path_parameters.get('name')
                if not name:
                    return {
                        'statusCode': 400,
                        'headers': headers,
                        'body': json.dumps({'error': 'Name parameter is required'})
                    }

                customer = get_customer_by_name(name)
                
                if not customer:
                    return {
                        'statusCode': 404,
                        'headers': headers,
                        'body': json.dumps({'error': 'Customer not found'})
                    }
                
                orders = get_customer_orders(customer['customerid'])
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'customer': customer,
                        'orders': orders
                    })
                }
        
        return {
            'statusCode': 405,
            'headers': headers,
            'body': json.dumps({'error': 'Method not allowed'})
        }
        
    except Exception as e:
        if 'messageVersion' in event:
            # Bedrock agent error response
            return {
                'messageVersion': '1.0',
                'response': {
                    'actionGroup': event.get('actionGroup', ''),
                    'apiPath': event.get('apiPath', ''),
                    'httpMethod': 'GET',
                    'httpStatusCode': 500,
                    'responseBody': {
                        'application/json': {
                            'body': json.dumps({'error': str(e)})
                        }
                    }
                }
            }
        else:
            # API Gateway error response
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            } 