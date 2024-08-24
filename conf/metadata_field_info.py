from langchain.chains.query_constructor.base import AttributeInfo

'''
    流程图self-query检索
'''
flowchart_metadata_field_info = [
    AttributeInfo(
        name="name",
        description="流程图的名字",
        type="string",
    ),
    AttributeInfo(
        name="disease",
        description="疾病的名字",
        type="string",
    ),
    AttributeInfo(
        name="symptoms",
        description="症状的名字",
        type="string or list[string]",
    )
]

dd_book_metadata_field_info = [
    AttributeInfo(
        name="symptom",
        description="疾病的名字",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="标题",
        type="string",
    ),
    AttributeInfo(
        name="type",
        description="类型",
        type="string",
    )
]