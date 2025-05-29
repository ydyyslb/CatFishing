import pandas as pd
from neo4j import GraphDatabase
import logging
import sys

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jImporter:
    def __init__(self, uri, user, password):
        try:
            logger.info(f"Connecting to Neo4j at {uri}")
            if not test_connection(uri, user, password):
                logger.error("Failed to connect to Neo4j. Exiting program.")
                sys.exit(1)
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.error("Please check:")
            logger.error("1. Is Neo4j running?")
            logger.error("2. Are the credentials correct?")
            logger.error("3. Is the URI correct?")
            sys.exit(1)
    
    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def import_data_from_excel(self, excel_file):
        try:
            logger.info(f"Reading Excel file: {excel_file}")
            df = pd.read_excel(excel_file)
            logger.info(f"Successfully read Excel file. Found {len(df)} rows")
            
            with self.driver.session() as session:
                for index, row in df.iterrows():
                    try:
                        session.execute_write(self._add_data, row)
                        if (index + 1) % 100 == 0:
                            logger.info(f"Processed {index + 1} rows")
                    except Exception as e:
                        logger.error(f"Error processing row {index + 1}: {str(e)}")
                        continue
            logger.info("Data import completed successfully")
        except Exception as e:
            logger.error(f"Error during data import: {str(e)}")
            raise
    
    def _add_data(self, tx, row):
        try:
            choices_text = row['Choices'].split(', ') if row['Choices'] else []

            data = {
                'grade': row['Grade'],
                'subject': row['Subject'],
                'topic': row['Topic'],
                'category': row['Category'],
                'skill': row['Skill'],
                'task': row['Task'],
                'question': row['Question'],
                'solution': row.get('Solution', None),
                'lecture': row.get('Lecture', None),
                'choices': choices_text,
                'correctIndex': int(row['Answer']) if pd.notnull(row['Answer']) else None
            }
            if pd.isna(row.get('Lecture')):
                data['lecture'] = None
                
            query = """
            MERGE (g:Grade {name: $grade})
            MERGE (s:Subject {name: $subject})
            MERGE (g)-[:INCLUDES]->(s)
            MERGE (t:Topic {name: $topic})
            MERGE (t)-[:BELONGS_TO]->(s)
            MERGE (c:Category {name: $category})
            MERGE (c)-[:PART_OF]->(t)
            MERGE (sk:Skill {name: $skill})
            MERGE (c)-[:REQUIRES]->(sk)
            MERGE (task:Task {type: $task})
            CREATE (q:Question {text: $question})
            SET q.solution = COALESCE($solution, '')
            MERGE (q)-[:DEMONSTRATES]->(sk)
            MERGE (q)-[:IS_A]->(task)
            WITH q, $choices as choices, $correctIndex as correctIndex
            UNWIND range(0, size(choices) - 1) as i
            MERGE (ch:Choice {text: choices[i], questionId: id(q), isCorrect: i = correctIndex})
            MERGE (q)-[:HAS_CHOICE]->(ch)
            FOREACH (_ IN CASE WHEN $lecture IS NOT NULL THEN [1] ELSE [] END | 
                MERGE (l:Lecture {content: $lecture})
                MERGE (q)-[:EXPLAINED_BY]->(l))
            """
            tx.run(query, data)
        except Exception as e:
            logger.error(f"Error in _add_data: {str(e)}")
            raise

def test_connection(uri, user, password):
    try:
        logger.info(f"Attempting to connect to Neo4j with:")
        logger.info(f"URI: {uri}")
        logger.info(f"User: {user}")
        logger.info("Password: ******")  # 不显示实际密码
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # 尝试一个简单的查询来验证连接
            result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
            record = result.single()
            if record:
                logger.info(f"Successfully connected to Neo4j {record['name']} {record['edition']} Edition")
                logger.info(f"Version: {record['versions'][0]}")
                return True
            else:
                logger.error("Connection test failed: No response from server")
                return False
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        if "AuthenticationRateLimit" in str(e):
            logger.error("Too many failed authentication attempts. Please:")
            logger.error("1. Stop the Neo4j server")
            logger.error("2. Wait a few minutes")
            logger.error("3. Restart the Neo4j server")
            logger.error("4. Try again with the correct credentials")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Starting data import process")
        
        # 测试连接
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "123456"  # 请修改为你的实际密码
        
        logger.info("Testing Neo4j connection...")
        if not test_connection(uri, user, password):
            logger.error("Failed to connect to Neo4j. Please check your credentials and try again.")
            logger.error("Make sure:")
            logger.error("1. Neo4j server is running")
            logger.error("2. The credentials are correct")
            logger.error("3. The server is not locked due to too many failed attempts")
            sys.exit(1)
            
        logger.info("Connection successful, proceeding with data import...")
        importer = Neo4jImporter(uri, user, password)
        importer.import_data_from_excel("problems.xlsx")
        importer.close()
        logger.info("Data import process completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
