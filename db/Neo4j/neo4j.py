from conf import *
from py2neo import Node,Relationship,Graph,Path,Subgraph
from py2neo import NodeMatcher,RelationshipMatcher
# graph = Graph('neo4j://localhost:7687/', auth=('neo4j','123456'))
# Person2 = Node('Person', name='于一',age=1,male=True,relation="dog")    
# graph.create(Person2)  # 创建结点
# Person3 = Node('Person', name='杨聪',age=10,male=False,relation="pig")    
# graph.create(Person3)  # 创建结点
# relation16 = Relationship(Person2,'室友',Person3)
# graph.create(relation16)  # 创建关系
class Neo4j:
    def __init__(self, url, username, password) -> None:
        self.graph = Graph(url, auth=(username, password))
        self.node_matcher = NodeMatcher(self.graph)
        self.relation_matcher = RelationshipMatcher(self.graph)
        self.max_id = -1
        # self.init()

    def init(self):
        all_exists_nodes = self.get_all_nodes()
        for node in all_exists_nodes:
            if node.get('id')> self.max_id: self.max_id = node.get('id')

    def add_node(self, node_label, node_id, node_name, **kwargs):
        console.log(f"node label:{node_label}")
        if self.node_matcher.match(*node_label, **{"id": node_id}).exists():
            console.log('node exist!')
            new_node = self.node_matcher.match(*node_label, **{"id": node_id}).first()
            for key in kwargs.keys():
                new_node[key] = kwargs[key]
            self.graph.push(new_node)
        else:
            console.log(f"[blod blue] Add node: {node_name}")
            new_node = Node(*node_label, name=node_name, **kwargs)
            if kwargs['id'] > self.max_id: self.max_id = kwargs['id']
            self.graph.create(new_node)
        return new_node

    def add_dt_nodes(self, node_list):
        for node in node_list: 
            ...
    
    def create_relation(self, pre_node, relation, last_node, **kwargs):
        # console.log(f"create relation between {pre_node} and {last_node}")
        relation_check = list(self.relation_matcher.match([pre_node, last_node], r_type=None))
        if len(relation_check)!=0:
            for re in relation_check:
                self.graph.separate(re)
        print(f"create relation between: {pre_node['Name']} and {last_node['Name']}")
        self.graph.create(Relationship(pre_node, relation, last_node, **kwargs))

    def match_nodes(self, lable, **kwargs):
        print(kwargs)
        matched_nodes = self.node_matcher.match(lable, **kwargs).first()
        return matched_nodes
    
    def match_all_edges(self, nodes):
        print(f"match_all_edges")
        all_relations = list()
        added_ids = list()
        for node in nodes:
            relationships = list(self.relation_matcher.match([node], r_type=None))
            for relation in relationships:
                if relation.get("id") not in added_ids:
                    all_relations.append({
                        "id": relation.get("id"),
                        "source":relation.get("source"), 
                        "sourceNode": int(relation.get("sourceNode")), 
                        "target": relation.get("target"), 
                        "targetNode": int(relation.get("targetNode")),
                        "label": relation.get("label")
                    })
                    added_ids.append(relation.get("id"))
        return all_relations
    
    def edit_node(self, labels, name, **kwargs):
        edit_node = self.node_matcher.match(*labels, id=kwargs['id']).first()
        if edit_node==None:
            console.log('edit node is None')
            return False
        else:
            for key in kwargs:
                edit_node[key] = kwargs[key]
            self.graph.push(edit_node)
            return True
        
    def get_all_nodes(self):
        nodes = self.graph.run('MATCH (n) RETURN n LIMIT 10000').data()
        res_nodes = list()
        for node in nodes:
            res_nodes.append(node['n'])
        return res_nodes
    
    def delete_node(self, label, **kwargs):
        console.log(f"delete node:{kwargs}")
        console.log(f"delete node:{kwargs['id']}")
        delete_nodes = self.node_matcher.match(*label, **{'id': kwargs['id']}).all()
        console.log(f"all node delete: {delete_nodes}")
        if delete_nodes == None:
            return None
        for node in delete_nodes:
            self.graph.delete(node)
        return True

    def delete_label_all_data(self, label):
        all_nodes = self.node_matcher.match(*label).all()
        for node in all_nodes:
            relationships = list(self.relation_matcher.match([node], r_type=None))
            for rela in relationships:
                self.graph.separate(rela)
        for node in all_nodes:
            self.graph.delete(node)
            
    def delete_all(self):
        nodes = self.graph.run('MATCH (n) RETURN n LIMIT 10000').data()
        for node in nodes:
            console.log(f"delete node: {node['n']}")
            self.graph.delete(node['n'])

        console.log(f"Finish delete {len(nodes)} nodes!")
    def run_server(self):
        ...