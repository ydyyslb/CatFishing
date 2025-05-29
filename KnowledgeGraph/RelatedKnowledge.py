import pandas as pd
from neo4j import GraphDatabase
import os

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def import_relationships_from_csv(self, csv_file):
        df = pd.read_csv(csv_file, encoding='GBK')
        with self.driver.session() as session:
            for index, row in df.iterrows():
                            # 检查PreKnowledge列是否是NaN，是则使用空列表
                            if pd.isna(row['PreKnowledge']):
                                pre_knowledge_points = []
                            else:
                                pre_knowledge_points = str(row['PreKnowledge']).split('##')
                            for pk in pre_knowledge_points:
                                session.execute_write(self._add_relationship, row['Skill'], pk)
    
    def _add_relationship(self, tx, skill, pre_knowledge):
        query = """
        MATCH (skill:Skill {name: $skill})
        MATCH (preSkill:Skill {name: $pre_knowledge})
        MERGE (preSkill)-[:PREREQUISITE_FOR]->(skill)
        """
        tx.run(query, skill=skill, pre_knowledge=pre_knowledge)

# Example usage
if __name__ == "__main__":
    # Your local neo4j account password.
    importer = Neo4jImporter("bolt://localhost:7687", "neo4j", "123456")
    
    # Get the absolute path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "unique_topics_skills_and_categories.csv")
    
    # Import relationships from CSV
    importer.import_relationships_from_csv(csv_path)
    importer.close()
