ICL = {
    "类型": "条件判断",
    "描述": "转移性胃癌",
    "条件": "( cM 为 'M1' )",
    "子节点": [
        {
            "类型": "条件判断",
            "描述": "腹膜转移",
            "条件": "( 腹膜转移 为 1 )",
            "子节点": [
                {
                    "类型": "决策动作",
                    "决策内容": "诊断为：腹腔化疗"
                }
            ]
        },
        {
            "类型": "条件判断",
            "描述": "除外腹膜转移的其他部位远处转移",
            "条件": "( 腹膜转移 为 0 )",
            "子节点": [
                {
                    "类型": "决策动作",
                    "决策内容": "诊断为：一线化疗"
                }
            ]
        }
    ]
}
# 决策树自动补全的prompt
prompt_v1 = '''
给你一段医学文本和一颗有if-else组成的决策树，你需要根据医学文本补充决策树中的None部分，None可以是下一个if条件判断，也可以是最终的治疗方案，要求补充的决策树内容必须全部来自于给定的医学文本，不可以出现医学文本之外的内容，要求补充完后的决策树需要包含已知的决策树部分。
医学文本为：{text}，决策树为：{tree}
补充完成之后的决策树为：
'''
prompt_v2 = '''
给你一段医学文本和一颗由二叉树构成的决策树，你需要根据医学文本补充决策树中叶节点的子节点，即补充决策树中true_branch和false_branch为空的节点，子节点可以是下一个条件判断节点，也可以是最终的治疗方案。条件判断节点和决策方案的节点严格按照下面格式生成。要求只补充一层节点，不需要生成子节点的子节点。要求补充的决策树内容必须全部来自于给定的医学文本，不可以出现医学文本之外的内容，并且每个节点来自医书的哪几句话在source中给出。要求补充完后的决策树需要包含已知的决策树部分。
条件节点的表示方式如下所示：
{{
    "id": 1,
    "parent_id": 0,
    "nodeName": "非转移性胃癌",
    "type": "Condition",
    "conditions": "( cM 为 'M0' )",
    "source": "",
    "true_branch": {{}},
    "false_branch": {{}}
}}
其中id为当前节点的序号，parent_id为父节点的id，nodeName为条件节点的概述，conditions为需要判断的全部条件，type为节点类型，条件节点为Condition，source为节点的出处，来自医疗文本的哪几句话，true_branch和false_branch对应了conditions判断为True和False的分支。
决策方案节点的表示方式如下：
{{
    "id": 1,
    "parent_id": 0,
    "type": "Decision",
    "Action": "非转移性胃癌",
    "source": ""
}}
其中id为当前节点的序号，parent_id为父节点的id，type为节点类型，决策方案节点的type为Decision，Action为具体的决策方案，source为节点的出处，来自医疗文本的哪几句话。决策方案节点为决策树的叶子节点，没有子节点。

医学文本为：{text}，决策树为：{tree}

要求给出的决策树中每个节点的true_branch和false_branch之生成一个子节点。
补充完成之后的决策树为：
'''


prompt_v3 = '''
给你一段医学文本和一颗由二叉树构成的决策树的一条路径，你需要根据医学文本补充决策树路径的下一个节点，下一个节点可以是下一个条件判断节点，也可以是最终的治疗方案。条件判断节点和决策方案的节点严格按照下面格式生成。要求只生成一个节点。要求补充的决策树内容必须全部来自于给定的医学文本，不可以出现医学文本之外的内容，并且每个节点来自医书的哪几句话在source中给出。要求补充完后的决策树需要包含已知的决策树部分。
条件节点的表示方式如下所示：
{{
    "id": 1,
    "parent_id": 0,
    "nodeName": "非转移性胃癌",
    "type": "Condition",
    "conditions": "( cM 为 'M0' )",
    "source": "",
    "true_branch": {{}},
    "false_branch": {{}}
}}
其中id为当前节点的序号，parent_id为父节点的id，nodeName为条件节点的概述，conditions为需要判断的全部条件，type为节点类型，条件节点为Condition，source为节点的出处，来自医疗文本的哪几句话，true_branch和false_branch对应了conditions判断为True和False的分支。
决策方案节点的表示方式如下：
{{
    "id": 1,
    "parent_id": 0,
    "type": "Decision",
    "Action": "非转移性胃癌",
    "source": ""
}}
其中id为当前节点的序号，parent_id为父节点的id，type为节点类型，决策方案节点的type为Decision，Action为具体的决策方案，source为节点的出处，来自医疗文本的哪几句话。决策方案节点为决策树的叶子节点，没有子节点。

医学文本为：{text}，决策树为：{tree}

要求只生成一个子节点, 生成的节点不能包含有子节点
补充完成之后的决策树为：
'''

prompt_v4 = '''
给你一段医学文本和一条由条件节点组成的决策过程，你需要根据医学文本该决策路径的下一个节点，下一个节点可以是下一个条件判断节点，也可以是最终的治疗方案。条件判断节点和决策方案的节点严格按照下面格式生成。要求只生成一个节点。要求补充的决策树内容必须全部来自于给定的医学文本，不可以出现医学文本之外的内容，并且每个节点来自医书的哪几句话在source中给出。要求补充完后的决策树需要包含已知的决策树部分。
条件节点的表示方式如下所示：
{{
    "id": 1,
    "parent_id": 0,
    "nodeName": "非转移性胃癌",
    "type": "Condition",
    "conditions": "( cM 为 'M0' )",
    "source": ""
}}
其中id为当前节点的序号，parent_id为父节点的id，nodeName为条件节点的概述，conditions为需要判断的全部条件，type为节点类型，条件节点为Condition，source为节点的出处，来自医疗文本的哪几句话，true_branch和false_branch对应了conditions判断为True和False的分支。
决策方案节点的表示方式如下：
{{
    "id": 1,
    "parent_id": 0,
    "type": "Decision",
    "Action": "非转移性胃癌",
    "source": ""
}}
其中id为当前节点的序号，parent_id为父节点的id，type为节点类型，决策方案节点的type为Decision，Action为具体的决策方案，source为节点的出处，来自医疗文本的哪几句话。决策方案节点为决策树的叶子节点，没有子节点。

医学文本为：{text}，决策树为：{tree}

当{condition}判断为{tag}时的下一个节点为：
'''

prompt_v5 = '''
给你一段医学文本和一条由条件节点组成的决策过程，你需要根据医学文本该决策路径的下一个条件节点。条件判断节点严格按照下面格式生成。要求只生成一个节点。要求补充的决策树内容必须全部来自于给定的医学文本，不可以出现医学文本之外的内容，并且每个节点来自医书的哪几句话在source中给出。要求补充完后的决策树需要包含已知的决策树部分。
条件节点的表示方式如下所示：
{{
    "id": 1,
    "parent_id": 0,
    "nodeName": "非转移性胃癌",
    "type": "Condition",
    "conditions": "( cM 为 'M0' )",
    "source": ""
}}
其中id为当前节点的序号，parent_id为父节点的id，nodeName为条件节点的概述，conditions为需要判断的全部条件，type为节点类型，条件节点为Condition，source为节点的出处，来自医疗文本的哪几句话

医学文本为：{text}，决策树为：{tree}

当{condition}判断为{tag}时的下一个条件节点为：
'''

prompt_v6 = '''
假如你是一名医生，给你一段医学文本和一条由条件节点组成的决策过程，你需要根据医学文本该决策路径的下一个节点。下一个节点可以是条件节点也可以是动作节点。节点严格按照下面格式生成。要求只生成一个节点，要求补充的决策树内容必须全部来自于给定的医学文本，不可以出现医学文本之外的内容，并且每个节点来自医书的哪几句话在source中给出。要求补充完后的决策树需要包含已知的决策树部分。
节点的表示方式如下所示：
{{
    "id": 1,
    "parent": 0,
    "nodeName": "",
    "type": "",
    "details": "",
    "source": "" 
}}
其中id为当前节点的序号，parent为父节点的id，nodeName为条件节点的概述，
type为节点类型，分为表示判断条件的条件节点和表示治疗方案的动作节点，条件节点为Condition，动作节点为Action，
details为三元组格式表示的判断条件，例如：('肿瘤部位' 为 '胃')、('肿瘤直径' 大于 '2cm')，
如果节点type为Action，则details表示具体的治疗方案，
source为生成该节点的出处，来自医疗文本的哪几句话

details不能为空，如果无法生成下一个节点则返回None
医学文本为：{text}，决策树为：{tree}

在这条决策路径上，当{tag}条件{condition}时的下一个条件节点为：
'''
# 当{condition}判断为{tag}时的下一个条件节点为：
prompt = prompt_v2


# 校验部分的prompt
examine_prompt = ''''
校验标准：
1、
假如你是一名医生，请你判断以下决策路径是否符合上述的校验标准，如果不符合，给出修改意见
决策路径为：{decision_path}
校验结果为:
'''


transform_prompt = '''
你的任务是将由if-elif-else结构表示的决策树，转成由以下格式表示的条件节点和叶子节点，条件节点的格式为：
Condition(
    id=1,
    parent_id=0,
    description="非转移性胃癌",
    conditions="( cM 为 'M0' )"
)
其中id为当前节点的序号，parent_id为父节点的id，description为条件节点的概述，conditions为需要判断的全部条件
表示决策的叶子节点的格式为：
Action(  
    id=1,
    parent_id=0,
    action="内镜治疗"
)
其中id为当前节点的序号，parent_id为父节点的id，如果节点为None则忽略，最终结果以数组列表的格式返回！
下面会给你一颗由if-elif-else表示的决策树，以及几个已知的节点，你需要按照上面的要求，完成剩余节点的转换，将决策树转换成上述两种格式的节点。
if-elif-else表示的决策树如下：
{tree}
已知的的前几个节点为：
{nodes}
转换后的结果为：
'''

verify_flowchart_prompt = '''你是一个决策树节点抽取器，下面给你一个节点的介绍以及与该节点相关的医学文本，你需要根据医学文本，将该节点转化成决策树中的条件节点或动作节点。如果是条件节点，你需要以三元组的格式抽取出节点的判断条件；如果是动作节点，你需要抽取出节点的决策方案。抽取出的节点按照以下Json格式返回。
{{
    "id": 1,
    "parent": 0,
    "nodeName": "",
    "type": "",
    "details": "",
    "source": "" 
}}
其中id为当前节点的序号，parent为父节点的id，nodeName为条件节点的概述，
type为节点类型，分为表示判断条件的条件节点和表示治疗方案的动作节点，条件节点为Condition，动作节点为Action，
details为三元组格式表示的判断条件，例如：('肿瘤部位' 为 '胃')、('肿瘤直径' 大于 '2cm')，如果节点type为Action，则details表示具体的治疗方案。
source为生成该节点的出处，来自医疗文本的哪几句话
节点介绍为：{node}，医学文本为：{text}，将该节点转化成决策树中的条件节点或动作节点之后的结果为：'''

transform_flowchart_prompt = '''你是一个决策树节点转换器，你需要补充决策树中节点的详细判断条件或者更加详细的决策方案。下面给你决策树中的一个节点的内容，该节点的父节点和子节点的内容，以及这三个节点相关的医学文本。 你需要根据医学文本和该节点与其父节点和子节点的关系，将该节点转化成决策树中的具有明确判断条件的条件节点或者具有详细决策方案的动作节点。如果是条件节点，你需要给出明确的判断条件；如果是动作节点，你需要抽取出节点的决策方案。抽取出的节点按照以下Json格式返回。
{{
    "type": "",
    "details": "",
    "source": "" 
}}
其中type为节点类型，分为表示判断条件的条件节点和表示治疗方案的动作节点，条件节点为Condition，动作节点为Action。
如果节点type为Condition，details为判断条件，如果节点type为Action，则details表示具体的治疗方案。
source为生成该节点的依据，你需要对为什么这么转换进行解释，最好引用医书文本的原文进行解释。
需要转换的节点内容为：{node}，该节点的父节点为：{parent_node}，该节点的子节点为：{child_node}，相关的医学文本为：{text}，将该节点转化成决策树中的条件节点或动作节点之后的结果为：'''

node_type_classify_prompt = '''你是一个节点类型判断器，下面会给你一个与医学相关的节点，你需要判断给出的它是条件节点还是动作节点。条件节点就是包含有需要判断的条件信息，动作节点则是表示需要进行的动作，可能是医疗过程中的某一个操作，也可能是一种治疗方案，还可能是某一种诊断出的疾病。如果是条件节点则返回Condition，如果是动作节点则返回Action。节点为：{node}，节点类型为：'''

condition_node_transform_prompt = '''你是一个条件节点增强器，给你决策树中的一个条件节点，你需要对该节点进行数据增强，根据给出的医书文本补充该条件节点的详细判断条件。 你需要根据医学文本以及这些节点之间的的关系，完善该条件节点的判断条件的条件节点。并且要求给出这个条件节点在医学文本中的依据。抽取出的节点按照以下Json格式返回。
{{
    "details": "",
    "source": "" 
}}
其中details为判断条件，如果节点type为Action，则details表示具体的治疗方案。
source为生成该条件节点的依据，你需要抽取出医学文本中能够佐证这个条件节点的内容。
需要转换的条件节点为：{node}，相关的医学文本为：{text}，对该条件节点进行增强后的结果为：'''

action_node_transform_prompt = '''你是一个决策树动作节点增强器，给你决策树中的一个与医学相关的动作节点，你需要对该动作节点的内容进行补充，补充的内容可能是医学过程中的一步操作，也可能是一个详细的治疗方案，还可能是诊断 出的最终疾病，也就是决策树最终的输出结果。你需要根据医学文本和该节点与其父节点的关系，将该节点转化成决策树中具有详细决策方案的动作节点。并且要求给出这个条件节点在医学文本中的依据。抽取出的节点按照以下Json格式返回。
{{
    "details": "",
    "source": ""
}}
其中details为判断条件，如果节点type为Action，则details表示具体的治疗方案。
source为生成该节点的依据，你需要抽取出医学文本中能够佐证这个条件节点的内容。
需要转换的节点内容为：{node}，相关的医学文本为：{text}，对该动作节点进行增强后的结果为：'''

index_search_prompt = '''你是一个医疗决策树相关文本检索器，你的任务是根据给出的决策树中一个节点的内容，从向量数据库中检索出与该节点相关性最高的3段不同的文本。要求检索出的文本是与医疗诊断或医疗决策相关的内容，并且其中包含有与给出的节点内容相关的条件信息、治疗方案相关信息或者可能的下一步操作相关的内容。
检索出的3段文本使用换行符进行分隔
给出的决策树中一个节点的内容为：{question}'''

index_search_prompt_v2 = '''你是一个医疗决策树相关文本检索器，你的任务是根据给出的决策树中几个节点内容，从向量数据库中检索出与当前节点及其父节点和子节点相关性最高的一段文本。要求检索出的文本是与医疗诊断或医疗决策相关的内容，并且其中包含有与给出的节点内容相关的条件信息、治疗方案相关信息或者可能的下一步操作相关的内容。要求能够根据检索出的文本从父节点推理出当前节点，能从当前节点推理出当前节点的子节点。\n{question}'''

get_disease_name_prompt = '''下面给你一本医书的内容，请你告诉我这本书主要与哪些疾病有关。医书内容为：{book_content}，书中主要提及的疾病为：'''

get_symptons_prompt = '''下面给你一本医书的内容，你需要抽取出这本书中提到的疾病的主要症状。提取出的全部症状以数组的格式返回，例如：["发烧", "头疼"]。医书内容为：{book_content}，抽取出的全部症状为：'''


flowchart_search_prompt = '''你是一个向量数据库的检索器，向量数据库中存储着按以下JSON格式表示的节点，多个节点组成的数组表示了一棵决策树：
{{
    "id": 1,
    "parent": 0,
    "content": "有无呼吸衰竭?",
    "label": "有"
    "source": ""
    "childs": []
}}
其中content为节点的主要内容，label为节点的标签，source为节点出处。
下面给你几个节点的content部分的内容，你的任务是从向量数据库中检索出包含有这几个节点的3棵不同的决策树。
要求给定的这个几个节点内容应该与检索出的决策树中某一个节点的content部分内容相同或者相似；
要求检索出的决策树必须包含给定的几个节点的内容。
检索出的3棵不同的决策树使用换行符进行分隔
给出的几个决策树节点为：{question}'''



generate_dialog_prompt = '''你是一名医生，现在给你一个患者在对话中给出的信息，以及一棵与患者所述疾病相关的决策树信息，你需要根据决策树信息和患者信息，生成给患者的回复。
决策树信息为:{sub_tree}，患者提问的消息为：{dialog},给患者的回复：
'''

from langchain.prompts import PromptTemplate

check_action_prompt = PromptTemplate(template='''''', input_variables=[])
action_augment_prompt = PromptTemplate(template='''''', input_variables=[])
check_condition_prompt = PromptTemplate(template='''''', input_variables=[])
condition_augment_prompt = PromptTemplate(template='''''', input_variables=[])

transform_node_prompt_ch = '''请对下面条件语句进行转换，要求将条件和结果整合成一句话。

input：条件：重复尿试纸测试（几天后）结果：阳性
output：几天后，重复尿试纸测试为阳性

input：条件：分泌物增加 结果：是
output：存在分泌物增加

input：条件：是否绝经后 结果：否
output：不是在绝经后

input：条件：远看时图像之间的距离最大 结果：否
output：远看时图像之间的距离不是最大

input：条件：体重减轻是刻意的还是非刻意的  结果：刻意的
output：体重减轻是刻意的

input：条件：妊娠 结果：是
output：是处在妊娠期

input：条件：有预警症状吗？预警症状为：发热、血便、从睡中觉醒、结肠癌家族病史、免疫功能低下 结果：否
output：没有出现发热、血便、从睡中觉醒、结肠癌家族病史或免疫功能低下的情况

input：条件：是否有神经损害的疾病证据 结果：是
output：有神经损害的疾病证据

input：条件：{condition} 结果：{label}
output：'''

transform_node_prompt_en = '''Here's the translation of the text into English:

"Please transform the following conditional statements by integrating the condition and the result into one sentence.

input: Condition: Repeat urine dipstick test (in a few days). Result: Positive
output: In a few days, the repeat urine dipstick test is positive.

input: Condition: Increased secretions. Result: Yes
output: There is an increase in secretions.

input: Condition: Is it post-menopausal? Result: No
output: It is not post-menopausal.

input: Condition: Maximum distance between images when viewed from afar. Result: No
output: The distance between images when viewed from afar is not the greatest.

input: Condition: Is weight loss intentional or unintentional? Result: Intentional
output: The weight loss is intentional.

input: Condition: Pregnancy. Result: Yes
output: It is during pregnancy.

input: Condition: Are there any warning symptoms? Warning symptoms include: fever, bloody stools, waking from sleep, family history of colon cancer, immunodeficiency. Result: No
output: There are no symptoms such as fever, bloody stools, waking from sleep, family history of colon cancer, or immunodeficiency.

input: Condition: Is there evidence of nerve damage? Result: Yes
output: There is evidence of nerve damage.

input: Condition: {condition} Result: {label}
output: '''

node_label_transform_prompt = PromptTemplate(template=transform_node_prompt_en, input_variables=["condition", "label"])

patient_answer_content_prompt = PromptTemplate(template='''请对下面条件语句进行转换，要求将条件和结果整合成一句话。

input：条件：重复尿试纸测试（几天后）结果：阳性
output：几天后，重复尿试纸测试为阳性

input：条件：分泌物增加 结果：是
output：存在分泌物增加的现象

input：条件：是否绝经后 结果：否
output：不是在绝经后

input：条件：远看时图像之间的距离最大 结果：否
output：远看时图像之间的距离不是最大

input：条件：体重减轻是刻意的还是非刻意的  结果：刻意的
output：体重减轻是刻意的

input：条件：妊娠 结果：是
output：是处在妊娠期

input：条件：有预警症状吗？预警症状为：发热、血便、从睡中觉醒、结肠癌家族病史、免疫功能低下 结果：否
output：没有出现发热、血便、从睡中觉醒、结肠癌家族病史或免疫功能低下的情况

input：条件：评估是否有预警症状：剧烈疼痛、视力下降异物感和畏光 结果：否
output：没有剧烈疼痛、视力下降异物感或者畏光的症状

input: 条件：明显眶周肿胀、转动眼球时会痛吗？ 结果：是
output：有出现明显眶周肿胀或者转动眼球时会痛中的一个症状

input：恶心、畏光、畏声、单侧头痛和悸动性头痛 结果：是
output：有恶心、畏光、畏声、单侧头痛或者悸动性头痛中的一种或几种症状

input：条件：{condition} 结果：{label}
output：''', input_variables=["condition", "label"])

decision_tree_node_infer_prompt = """下面给你一个患者的主诉和一个判断条件，你需要判断患者的主诉是否满足所给的条件，如果条件中锁提及的症状在主诉中有出现，并且满足条件中的要求，则返回True，否在返回False。注意，患者没有提及的症状，不能判断为患者没有该症状。

患者主诉为：我目前正在住院治疗，最近没有出国旅行，但我一直发热
Condition：正在发热
Output：True

患者主诉为：我最近一直感到眩晕
Condition：正在经历眩晕
Output：True

患者主诉为：我夜间存在盗汗，并且没有发热，同时还出现了预警症状和体征
Condition：同时存在新发高血压、新发头痛、阵发性症状潮红、腹泻和喘息。
Output：False

患者主诉为：我最近咽部疼痛，但没有化脓性并发症的预警症状
Condition：不存在化脓性并发症的预警症状
Output：True

患者主诉为：医生，我最近一直感到恶心和呕吐，但排除了妊娠的可能性，而且这是一个急性的情况
Condition：不是全身症状
Output：False

患者主诉为：请问医生，我出现头晕和周围物体不停旋转的症状，这可能是什么病引起的？，
Condition：周围物体不停地旅转，仿佛刚刚走下旋转木马，
Output：True

患者主诉为：{Q}，
Condition：{condition}，
Output："""

# 患者主诉：我在夜间出现盗汗的症状，没有发热、预警症状和绝经期症状，也没有使用可疑药物
# Condition：考虑胃食管反流，
# Output：False

"""
作为裁判，评估一下以下对于同一患者问题的两个回答。

患者问题：
{Question}
回答1：
{Answer1}
回答2：
{Answer2}

评价标准按优先顺序依次为逻辑性、主动性、准确性、有用性，具体定义如下：
可解释性：医生的追问和诊断结果是有合理依据的。
逻辑性：医生的诊断或者追问具有逻辑，多轮对话的顺序是循序渐进的。
准确性：医生提供的诊断或建议是准确的，没有事实错误。 结论是不是随意做出的。
主动性：当信息不充分时，医生可以主动、明确地要求患者提供更多信息。
有用性：医生可以为患者提供明确、指导性和实用的帮助，解决患者的疑虑。

注意：
按照**可解释性>逻辑性>准确性>主动性>有用性**的重要性进行评估，如果有冲突，优先考虑前者。
输出格式：
根据以上标准，判断“回答l”相对于“回答2”的结果。 输出为：赢、输。
"""


dialgnose_framework_prompt = '''下面给你一些与“{chief}”诊断相关的信息，你需要将这些信息整合进下面给出的问诊框架中，使问诊框架更加符合“{chief}”的问诊过程。要求原本框架的步骤不能改变，不能增加和删除步骤，每个步骤的也需要与原本框架意思相同。
注意：不需要给出示例。

与“{chief}”相关的信息为：
{content}

问诊框架为：
第一步：向患者打招呼，简短自我介绍，确认患者的主诉信息，并进行开放式提问，引导患者讲述更多自己的症状。
第二步：继续开放式提问，引导患者讲述更多自己的症状、问题、病史等。
第三步：简单总结患者提及的症状、现病史等信息，针对性提问患者的其他症状信息。

整合后的问诊框架为：'''

patient_question_init_prompt_ch = '''你是一名患者，你现在正在与医生进行对话，你需要向医生描述自己的症状，并咨询医生自己得了什么病。
提出的问题必须满足以下几点要求：
1、你的主要症状为：{symptoms}，要求必须基于此症状信息提出问题；
2、不能包含专业名词，要符合患者作为普通人的说话习惯，如果给出的症状信息中包含有专业名词，你需要根据该名词的释义来描述自己的症状；
3、要求以患者的口吻进行提问，问题简洁自然；

提出的问题为：'''

patient_question_init_prompt_en = '''You're a patient, currently having a conversation with a doctor. You need to describe your symptoms to the doctor and inquire about their diagnosis. 

The question you ask must meet the following requirements:
1. Your main symptoms are: {symptoms}, and your question must be based on this symptom.
2. It should not contain medical jargon and should align with the speaking habits of an ordinary person. If the symptom information provided includes medical terms, you need to describe your symptoms based on the meaning of those terms.
3. The question should be asked in the patient's tone, concise and natural.

The question to be asked is:'''

patient_question_init_prompt = patient_question_init_prompt_en

patient_answer_with_tree_prompt = '''你是一名患者，你现在正在与医生进行对话，你需要根据下面给出的症状信息来回答医生提出的问题。
生成的回答必须满足以下几点要求：
1、本轮对话中，你需要讲出与{content}相关的症状，不能直接回答专业名词，需要回答与专业名词相关的症状；
2、要求以患者的口吻进行回答，并且语气友好，回答简洁自然；
3、不能出现专业性词语、专业术语，不能包含过于学术性的词语，回答要符合普通患者的说话习惯；
4、要求回答必须依据给出的症状信息，如果给出的症状信息有多个，则只需要回答一个即可；
5、如果无法根据自己的症状进行回答，则回答我不知道；
6、必须正面回答医生的问题，不能回答与医生问题无关的内容；
7、不要重复出现“你好”、“感谢”这样的问候语，回答不能过于冗长。
8、回复中不能有与前面对话重复的内容。
9、回复中不能重复出现感谢用语，也不用重复喊医生。
10、用英语回答。

与医生的对话为：
{dialog}
患者：'''

patient_answer_with_tree_prompt_v2 = '''你是一名患者，你现在正在与医生进行对话，下面给出你与医生的历史对话，你需要根据对话内容和下面给出的症状信息来回答医生提出的问题。
生成的回答必须满足以下几点要求：
1、本轮对话中，你需要依据这些信息来回答医生的问题：{content}。如果这些信息中包含有专业的医学名词，你需要根据名词的释义进行回答；
2、要求以患者的口吻进行回答，并且语气友好，简洁自然，回答中不能包含专业名词；
3、与患者的对话中不要出现重复的用语或内容。
4、生成的回复要求简洁，直白，不能一次性回复过多内容。

与医生的对话为：
{dialog}
患者：'''

patient_answer_with_tree_prompt_en = '''You are a patient, currently engaged in a conversation with a doctor. Below is a history of your past conversation with the doctor. Based on the content of the conversation and the symptoms provided below, you need to answer the doctor's questions.

Your answers must meet the following requirements:
1. In this round of conversation, you need to answer the doctor's questions based on this information: {content}. If there are any medical terms in this information, you should answer according to their definitions;
2. Your responses should be in the patient's tone, friendly, concise, and natural, without using medical jargon;
3. You must directly address the doctor's questions and avoid irrelevant content;
4. Avoid repeating phrases or content in your conversation with the doctor;
5. The response should be concise and straightforward, without too much content at once.

The conversation with the doctor is as follows:
{dialog}
Patient: '''

patient_answer_with_tree_prompt= patient_answer_with_tree_prompt_en

patient_answer_refine_prompt = '''下面给你一个患者与医生之间的对话，你需要按照以下几点标准来改写患者最后的回复内容。
标准为：
1. 必须保证改写后的意思不变，患者的意图也不变；
2. 患者的回复在简洁的基础上，需要足够口语化；
3. 患者的回复不能出现专业性词语、专业术语；
4. 患者的回复需要足够自然、没有语句不通顺的情况。
5. 患者的回复清晰的表达了自己的症状和问题。
6. 患者的回复没有与前面对话重复的内容。
7. 患者的回复不能重复出现感谢用语，也不用重复喊医生。

患者与医生的对话为：
{dialog}

改写后患者最后的回复为：'''

doctor_answer_no_framework = '''你是一名医生，下面给你一段你与患者的对话，你需要根据对话内容和以下要求，生成新的回复。
你生成的回复需要满足以下要求：
1、本次对话你只需要提问患者是否有与“{content}”相关的症状；
2、要求以医生的口吻生成回复，回复内容要求逻辑清晰，思路明确，对话过程连贯自然，不能出现对话中重复的内容，不包含需要专业知识才能理解的专业词汇；
3、要求一次最多向患者提出两个问题；
4、不要重复出现“你好”、“感谢”、“谢谢”这样的礼貌用语；
5、要求不能重复出现“明白了，了解了”、“这些症状可能会帮助我们更好地了解您的病情”、“感谢您的配合”或“希望您能告诉我更多细节，以便做出正确的诊断和治疗计划”相似的内容。

与患者的对话为：
{dialog}
医生：'''

doctor_answer_no_framework_v2 = '''你是一名医生，下面给你一段你与患者的对话，你需要根据对话内容和以下要求，生成新的回复。

你生成的回复需要满足以下要求：
1、在本轮对话中你只需要询问患者：{content}，不需要询问别的内容；
2、要求以医生的口吻生成回复，并且要求逻辑清晰，语气亲切自然，如果有医学名词，需要对其进行解释；
3、要求一次最多向患者提出两个问题；
4、与患者的对话中不要重复的用语或内容；
5、生成的回复要求简洁，直白，不能一次性回复过多内容。

与患者的对话为：
{dialog}
医生：'''

doctor_answer_no_framework_v3 = '''你是一名医生，你需要根据对话内容生成一个问题来询问患者：{content}。

你生成的回复需要满足以下要求：
1、不要询问与“{content}”无关的内容；
2、要求以医生的口吻生成回复，并且要求逻辑清晰，语气亲切自然，如果有医学名词，需要对其进行解释；
3、与患者的对话中不要重复的用语或内容；
4、生成的回复要求简洁，直白，不能一次性回复过多内容。

与患者的对话为：
{dialog}
医生：'''

doctor_answer_no_framework_en='''You are a doctor, and below is a segment of a conversation you are having with a patient. Based on the content of the conversation and the following requirements, you need to generate a new reply.

Your reply must meet these requirements:
1. You only need to generate a new reply based on this content: {content}, do not ask anything else;
2. The reply should be in the doctor's tone, clear in logic, and naturally cordial. If there are any medical terms, they need to be explained;
3. You may ask the patient up to two questions at a time;
4. Avoid repetitive language or content in your conversation with the patient;
5. The response should be concise and straightforward, without too much content at once.

The conversation with the patient is as follows:
{dialog}
Doctor: '''

doctor_answer_no_framework_en_v2 = '''You are a doctor, and you need to generate a question to ask the patient based on the content of the conversation: {content}.

Your response needs to meet the following requirements:
1. Do not ask about content unrelated to "{content}";
2. The response should be generated in the tone of a doctor, and it must be logical, courteous, and natural. If there are medical terms, they need to be explained;
3. Avoid repetitive language or content in the conversation with the patient;
4. The response should be concise, straightforward, and should not reply with too much content at once.

The conversation with the patient is:
{dialog}
Doctor: '''
doctor_answer_no_framework=doctor_answer_no_framework_en_v2

# 2、要求提出的问题在本来对话中需要引导患者说出与“{content}”相关的症状；
doctor_answer_with_framework_ch = '''你是一名医生，你现在正在与患者进行对话，你需要根据与患者的历史对话和问诊框架，生成新的回复。
你生成的问题需要满足以下要求：
1、本次对话中你应该{framework}；
2、你需要引导患者讲出与{content}相关的症状；
3、要求一次最多向患者提出两个问题；
4、要求以医生的口吻生成回复，回复内容要求逻辑清晰，思路明确，对话过程连贯自然，不能出现对话中重复的内容，不包含需要专业知识才能理解的专业词汇；
5、不要重复出现“你好”、“感谢”、“谢谢”这样的礼貌用语；
6、不出现与“这些症状可能会帮助我们更好地了解您的病情”、“感谢您的配合”或“希望您能告诉我更多细节，以便做出正确的诊断和治疗计划”相似的内容。

注意：一次最多只能提出两个问题！

与患者的对话为：
{dialog}
医生：'''

doctor_answer_with_framework_v2 = '''你是一名医生，下面给你一段你与患者的对话，你需要根据对话内容和以下要求，生成新的回复。

你生成的问题需要满足以下要求：
1、本次对话中，你必须：{framework}；
2、要求一次最多向患者提出两个问题，严禁询问患者是否有某个具体的症状；
2、要求以医生的口吻生成回复，并且要求逻辑清晰，语气亲切自然，如果有医学名词，需要对其进行解释；
4、与患者的对话中不要重复的用语或内容。
5、生成的回复要求简洁，直白，不能一次性回复过多内容。

与患者的对话为：
{dialog}
医生：'''

doctor_answer_with_framework_v3 = '''生成一个医生向患者提出的开放式问题，问题需要满足的条件为：
1、本次对话中，你必须：{framework}；
2、要求一次最多向患者提出两个问题，严禁询问患者是否有某个具体的症状；
与患者的对话为：
{dialog}
医生：'''

doctor_answer_with_framework_en = '''You are a doctor, and below is a segment of a conversation you are having with a patient. Based on the content of the conversation and the following requirements, you need to generate a new reply.

Your generated question must meet these requirements:
1. In this conversation, you must: {framework};
2. You may ask the patient up to two questions at a time. Avoid asking the patient about specific symptoms;;
3. The reply should be in the doctor's tone, clear in logic, and naturally cordial. If there are any medical terms, they need to be explained;
4. Avoid repetitive language or content in your conversation with the patient;
5. The response should be concise and straightforward, without too much content at once.

The conversation with the patient is as follows:
{dialog}
Doctor: '''

doctor_answer_with_framework_en_v2 = '''Generate an open-ended question that a doctor asks the patient. The question must meet the following criteria:

1. In this conversation, you can only: {framework}, and you cannot reply with any other content;
2. You must only pose one question, and you are strictly prohibited from asking the patient if they have a specific symptom.
3. Only ask one question.

The conversation with the patient is:
{dialog}
Doctor: '''
doctor_answer_with_framework = doctor_answer_with_framework_en_v2


doctor_final_answer_prompt_ch = '''你是一名医生，你正在与患者对话来诊断患者的疾病，通过对话能够诊断出患者的疾病为：{diagnose}，你需要根据历史对话，生成最终诊断结果的回复。

你生成的回复需要满足以下要求：
1、生成的回复要求逻辑清晰，思路明确，尽可能简洁直观，不啰嗦；；
2、首先需要回复患者根据患者的症状；
3、需回复患者诊断出来的疾病为：{diagnose}；
4、根据诊断出的疾病，给出相应的治疗建议；
5、要求以医生的口吻，并且口语化地进行回复，并且语气友好，以易于患者理解的方式进行提问；
6、要求不能出现对话中重复的内容，不要继续向患者提出问题；
7、不要重复出现“你好”、“感谢”、“谢谢”这样的礼貌用语；
8、要求生成的回复尽可能的简洁。
9、用英语回答。

与患者的对话为：
{dialog}
医生：'''

doctor_final_answer_prompt_v2 = '''你是一名医生，你正在与患者进行对话，你需要通过与患者对话来诊断患者的疾病，通过对话能够诊断出患者的疾病为：{diagnose}，你需要根据历史对话，生成最终诊断结果的回复。

你生成的回复需要满足以下要求：
1、生成的回复要求逻辑清晰，思路明确，尽可能简洁直观，不啰嗦；；
2、首先需要根据对话内容回复诊断出该疾病的依据，说明诊断逻辑；
3、需回复患者诊断出来的疾病为：{diagnose}；
4、要求以医生的口吻，并且口语化地进行回复，并且语气友好，以易于患者理解的方式进行提问；

与患者的对话为：
{dialog}
医生：'''

doctor_final_answer_prompt_en = '''"You are a doctor, currently in conversation with a patient. You need to diagnose the patient's illness through dialogue and determine that the patient's condition is {diagnose}. Based on the history of the conversation, you need to generate a response providing the final diagnosis.

Your response must meet the following requirements:
1. It should be logically clear, with a clear train of thought, as concise and straightforward as possible, avoiding unnecessary elaboration;
2. Firstly, you need to respond with the basis for diagnosing the illness based on the conversation content;
3. Confirm that the diagnosed illness is {diagnose};
4. The response should be in the doctor's tone, conversational, friendly, and phrased in a way that is easy for the patient to understand.

The conversation with the patient is as follows:
{dialog}
Doctor: '''

doctor_final_answer_prompt = doctor_final_answer_prompt_en

'''你是一名医生，你现在正在与患者进行对话，你需要根据历史对话和问诊框架，生成新的回复。
你生成的问题需要满足以下要求：
1、生成的回复要求逻辑清晰，思路明确，并且按照问诊框架中的步骤与患者进行对话；
2、要求一次最多向患者提出两个问题；
3、要求提出的问题在本来对话中需要引导患者说出与“{content}”相关的症状；
4、要求保证与患者的对话过程连贯自然，对话逻辑清晰；
5、要求以医生的口吻，并且口语化地进行询问，并且语气友好，不包含需要专业知识才能理解的专业词汇；
6、要求不能出现对话中重复的内容。
7、不要重复出现“你好”、“感谢”、“谢谢”这样的礼貌用语；
8、不出现与“这些症状可能会帮助我们更好地了解您的病情”、“感谢您的配合”或“希望您能告诉我更多细节，以便做出正确的诊断和治疗计划”相似的内容。

问诊框架为：
{framework}

与患者的对话为：
{dialog}
医生：'''


refine_generated_dialog_prompt = '''给定一段医生与患者之间的对话，你的任务是模仿示例，重写医生的回复，使对话更贴近真实场景下的问诊，重写完成后以原格式输出。医生在与患者对话的过程中，都应该使用口语化的表达进行简洁的回复，不要使用医学上的专业术语，不要重复之前已经提供的信息。
注意：只重写医生的回复。

示例1
需要重写的对话为：
[
    {{
        "type": "doctor",
        "msg": "您耳朵中是否有分泌物？除了耳朵疼、发热和上呼吸道感染的症状，您还有其他不适吗？"
    }},
    {{
        "type": "patient",
        "msg": "医生，我的耳朵没有分泌物。除了耳朵疼痛、发热和上呼吸道感染的症状，我没有其他不适。"
    }},
]
重写后的对话为：
[
    {{
        "type": "doctor",
        "msg": "您耳朵中是否有分泌物？除了耳朵疼、发热和上呼吸道感染的症状，您还有其他不适吗？"
    }},
    {{
        "type": "patient",
        "msg": "没有分泌物,我没有其他不适。"
    }},
]

示例2
需要重写的多轮对话为：
[
    {{
        "type": "doctor",
        "msg": "您好，我理解您目前出现了呼吸困难的症状。我想了解一下，除了呼吸困难之外，您是否还有其他不适的症状，比如咳嗽、胸痛、乏力等呢？"
    }},
    {{
        "type": "patient",
        "msg": "医生，除了呼吸困难以外，我没有咳嗽、胸痛或乏力等其他不适症状。只是每次呼吸都感到困难，很费劲。"
    }}
]
重写后的对话为：
[
    {{
        "type": "doctor",
        "msg": "您好，我理解您目前出现了呼吸困难的症状。我想了解一下，除了呼吸困难之外，您是否还有其他不适的症状，比如咳嗽、胸痛、乏力等呢？"
    }},
    {{
        "type": "patient",
        "msg": "我没有您提到的这些症状，只是每次呼吸都感到困难，很费劲。"
    }}
]

需要重写的对话
{Raw}
重写后的对话'''

refine_generated_dialog_prompt_v2 = '''给定一段医生与患者之间的对话，你的任务是模仿示例中医生的回复，重写医生的回复，使对话更贴近真实场景下的问诊，并且使医生的问诊过程更加符合下面给出的问诊框架，重写完成后以原格式输出。医生应该使用口语化的表达进行简洁的回复，不要使用医学上的专业术语，不要重复之前已经提供的信息。

问诊框架为：
{framework}

示例1
需要重写的对话为：
[
    {{
        "type": "patient",
        "msg": "医生，我最近老是头痛，请问这是怎么回事呢？"
    }},
    {{
        "type": "patient",
        "msg": "你好，我是医生。你最近头痛是最近才发生的吗，还是头痛症状与你之前所患的头痛症状不同了？可以告诉我头痛的具体感觉和持续时间吗？"
    }},
]
重写后的对话为：
[
    {{
        "type": "patient",
        "msg": "医生，我最近老是头痛，请问这是怎么回事呢？"
    }},
    {{
        "type": "doctor",
        "msg": "你好，我是医生，引起头痛的原因可能有很多，可以再具体讲讲你头痛的症状吗？头痛是最近才发生的吗？"
    }},
]

示例2
需要重写的多轮对话为：
[
    {{
        "type": "doctor",
        "msg": "您好，医生，我最近睡觉的时候总是出很多汗，我想知道这是因为什么导致的。"
    }},
    {{
        "type": "patient",
        "msg": "您好，我是您的医生。我了解您晚上总是盗汗，那除了盗汗之外，您还有没有其他与发热相关的症状呢？比如头痛、乏力、咳嗽等症状？盗汗可能有多种原因，我们需要了解更多细节才能做出正确的诊断。您能告诉我更多有关这些症状的情况吗？"
    }}
]
重写后的对话为：
[
    {{
        "type": "doctor",
        "msg": "您好，医生，我最近睡觉的时候总是出很多汗，我想知道这是因为什么导致的。"
    }},
    {{
        "type": "patient",
        "msg": "我知道你有盗汗的症状；你能和我讲讲盗汗时的具体情况吗?其他时间有出汗多的情况吗？"
    }}
]

注意：只重写医生的回复。

需要重写的对话
{Raw}

重写后的对话为：'''


refine_generated_dialog_prompt_v3 = '''给定一段医生与患者之间的对话，你的任务是根据问诊框架和给出的要求，重写医生的最后一次回复，使对话更贴近真实场景下的问诊，并且使医生的问诊过程更加符合下面给出的问诊框架和要求。

问诊框架为：
{framework}

同时，医生的回复需要满足以下要求：
1、要求与患者的对话逻辑清晰，思路明确，并且连贯自然；
2、要求一次最多向患者提出两个问题； 
3、要求以医生的口吻，并且口语化地进行询问，并且语气友好，不包含需要专业知识才能理解的专业词汇；
4、要求不能出现对话中重复的内容。
5、不要重复出现“你好”、“感谢”、“谢谢”这样的礼貌用语；
6、不出现与“这些症状可能会帮助我们更好地了解您的病情”、“感谢您的配合”或“希望您能告诉我更多细节，以便做出正确的诊断和治疗计划”相似的内容。


注意：只重写医生最后一次的回复。

医生和患者的对话为：
{Raw}

重写后的医生最后回复为：'''

refine_generated_dialog_prompt_v4 = '''给你一段医生与患者之间的对话，你的任务是重写对话中医生和患者的说话风格，使对话中医生和患者的回复分别满足下面几点要求。

对话中，医生的回复需要满足：
1、医生的回复需要使用口语化的语言，不能出现缩写，医学专有名词，都需要写成口语化的表达
2、医生的回复需要简洁，清晰。

对话中，患者的回复需要满足：
1、患者的回复需要是口语化，自然直白的表达，不能表现得过于专业。

注意：不能改变原本的对话顺序。

医生和患者的对话为：
{Raw}

重写后的医生和患者为：'''

refine_generated_dialog_prompt_v5 = '''Here is a conversation between a doctor and a patient. Your task is to rewrite the conversation in a way that the speaking styles of the doctor and the patient meet the following criteria.

For the doctor's replies:
1. The doctor's replies should use conversational language, without abbreviations or medical jargon, and should be written in a colloquial manner.
2. The doctor's replies should be concise and clear.

For the patient's replies:
1. The patient's replies should be in a natural and straightforward conversational style, avoiding overly professional expressions.

Note: Do not change the original order of the conversation.Return the result in JSON format.

The original conversation between the doctor and the patient is:
{Raw}

The rewritten conversation between the doctor and the patient is:'''


dialog_generation_once_prompt_v1 = '''下面给你一个医患诊断疾病的决策路口，你需要根据这个决策路径生成一段完整的医生和患者之间的对话，对话过程严格按照决策路径的顺序，决策路径中每个节点的content表示需要判断患者是否有该症状，label表示判断结果。最后一个节点表示诊断出的疾病。生成的对话需要以诊断出患者疾病结束。

注意：前两轮对话尽量以开放式问诊的方式，让患者自己描述症状。针对Urgent alarm symptoms节点，不要直接问患者是否有预警症状，而是应该让自己描述自己的症状，然后患者说出对应的预警症状。
生成的对话需要为英文，生成的对话可以有一点创造性，但是要求简洁、自然，符合医生和患者的对话习惯。

生成的对话要求以数组格式返回，格式要求如下：
[
    {{
        "type": "patient",
        "msg": ""
    }},
    {{
        "type": "doctor",
        "msg": ""
    }},
]

决策路径为：
{route}

生成的医患对话为：
'''

dialog_generation_once_prompt_v2 = '''下面给你一个医患诊断疾病的决策路径，你需要根据这个决策路径生成一段完整的医生和患者之间的对话，对话过程严格按照决策路径的顺序，决策路径中每个节点的content表示需要判断患者是否有该症状，label表示判断结果。最后一个节点表示诊断出的疾病。生成的对话需要以诊断出患者疾病结束。

生成的对话需要满足以下要求：
1、前两轮对话尽量以开放式问诊的方式，让患者自己描述症状。针对Urgent alarm symptoms节点，不要直接问患者是否有预警症状，而是应该让自己描述自己的症状，然后患者说出对应的预警症状；
2、医生和患者的对话不能与节点内容相同，需要进行改写；
2、生成的对话需要为英文；
3、生成的对话可以有一点创造性，但是要求简洁、自然，符合医生和患者的对话习惯。

生成的对话要求以数组格式返回，格式要求如下：
[
    {{
        "type": "patient",
        "msg": ""
    }},
    {{
        "type": "doctor",
        "msg": ""
    }},
]

决策路径为：
{route}

生成的医患对话为：
'''

dialog_generation_once_prompt='''You are a useful assistant. I will give you  a decision pathway for diagnosing a disease. And I need you to help me generate a dialogue between doctors and patients based on this decision pathway. I will give you different tips for each rounds. 
The generated dialogue must meet the following requirements:
1. The generated conversation can be slightly creative and must be concise, natural, and conform to the dialogue habits of doctors and patients;
2. The conversation between the doctor and the patient cannot be identical to the content of the nodes and needs to be rewritten;
3. The patient's responses must be colloquial, natural, and free of medical jargon. The doctor's responses should also be sufficiently colloquial, easy to understand, and patient-friendly.
4. Please pay attention to the final diagnosis made by the doctor. Briefly explain the reasons for diagnosing the disease and guide the patient to go to the hospital for further tests to confirm the diagnosis.
The following is an example. Please refer to the format.
Decision pathway：
Node 1: Fever
Node 2: Have you recently been in the hospital or traveled recently? - Yes,Recent travel abroad.
Dialogue:
[
    {{
        "type": "doctor",
        "msg": "Good morning. What brings you in today?"
    }},
    {{
        "type": "patient",
        "msg": "Good morning, doctor. I've been having a fever for the past few days."
    }},
    {{
        "type": "doctor",
        "msg": "I'm sorry to hear that. Have you recently been in the hospital or traveled anywhere recently?"
    }},
    {{
        "type": "patient",
        "msg": "Yes, I recently traveled abroad."
    }}
]

Decision pathway：
{route}
Dialogue:
'''

check_dialog_01_prompt = '''You need to determine whether the follow-up questions asked by the doctor in the following conversation with the patient are reasonable. Specifically, evaluate whether the questions posed by the doctor during the first three inquiries are appropriate. Your response should be either True or False. If there are any obviously unreasonable questions, answer True; otherwise, answer False.

The conversation between the doctor and the patient is as follows:
{dialog}

The judgment result is:
'''
translate_dialog_llm_prompt = '''Translate the following text between a doctor and a patient into Chinese:

The dialogue is:
{dialog}

The translation is:
'''


dialog_ultimate_prompt = """You are a medical dialogue optimizer. You need to optimize the following conversation between a doctor and a patient, making the expressions of both the doctor and the patient more colloquial and closer to the speaking habits of doctors and patients in real life.

Note: You are required to optimize only the expressions without changing the original sequence of the conversation. The final result must be in the same JSON format as the original conversation.

The conversation is:
{dialog}

The optimized conversation is:"""

augment_one_route_prompt = '''你是一个诊疗决策路径优化器。下面给出的决策路径存在有问诊信息不足的问题，医生通过询问患者获得的症状信息不足以诊断出患者的疾病，你需要根据下面给出的该疾病的症状表现信息，补充决策路径中的问诊节点，补充的内容主要为医生提出的问题和患者的回答。并且补充的内容需要满足以下几点要求：
1. 补充的内容不需要太多，足以诊断出患者的疾病即可。
2. 可以对原决策路径的节点进行改写，增加生活化的场景。

参考这个例子：

决策路径为：
[
    {{
        "content": "Headache"
    }},
    {{
        "content": "When did these headaches first start?Is this headache the same as ones you’ve had before or is it different in some way?",
        "label": "Old headache,( A chronic, benign, recurring headache)"
    }},
    {{
        "content": "Do you have any symptoms that occur at the same time as your headache?",
        "label": "No"
    }},
    {{
        "content": "Is the headache: - Piercing or sharp, as in an electric shock feeling?"
    }},
    {{
        "content": "Cluster headache"
    }}
]

优化之后的决策路径为：
[
    {{
        "content": "Head is very painful recently"
    }},
    {{
        "content": "When did these headaches first start?",
        "label": "It also appeared a month ago"
    }},
    {{
        "content": "Have you had a cold or fever recently",
        "label": "Yes"
    }},
    {{
        "content": "Have you experienced any stiffness in your neck or difficulty moving it?",
        "label": "Around my left eye, Always on the same side."
    }},
    {{
        "content": "This could be viral meningitis caused by the cold. However, don't worry, it will likely improve soon. Still, I recommend you go to the hospital for a detailed examination to confirm.",
        "label": "From 15 minutes to 3 hours"
    }},
    {{
        "content": "Do you have any symptoms that occur at the same time as your headache?",
        "label": "Red eye"
    }},
    {{
        "content": "Is the headache: - Piercing or sharp, as in an electric shock feeling?",
        "label": "Yes"
    }},
    {{
        "content": "Cluster headache"
    }}
]

根据以上要求、示例和下面给出的这个疾病的症状信息，你需要完成下面这条诊疗决策路径的优化，注意优化后的路径需要保持为英文。
该疾病的症状信息为：
{content}

决策路径为：
{route}

最终诊断的疾病
优化之后的决策路径为：
'''

augment_one_route_prompt_en = """You are a diagnostic decision path optimizer. The following decision-making path has the problem of insufficient consultation information. The symptom information obtained by the doctor by inquiring the patient is not enough to diagnose the patient's disease. You need to supplement the consultation node in the decision-making path according to the symptom and manifestation information of the disease given below. The supplementary content mainly includes the questions raised by the doctor and the patient's answers. 
Needs to meet the following requirements: 
1. The content of the supplement does not need to be too much, enough to diagnose the patient's disease. 
2. Nodes of the original decision path can be rewritten to add life-oriented scenes.
3. It needs to be returned in JSON structure

Consider this example: 

The decision path is:
[
    {{
        "content": "Headache"
    }},
    {{
        "content": "When did these headaches first start?Is this headache the same as ones you’ve had before or is it different in some way?",
        "label": "Old headache,( A chronic, benign, recurring headache)"
    }},
    {{
        "content": "Do you have any symptoms that occur at the same time as your headache?",
        "label": "No"
    }},
    {{
        "content": "Is the headache: - Piercing or sharp, as in an electric shock feeling?"
    }},
    {{
        "content": "Cluster headache"
    }}
]
The optimized decision path is as follows:
[
    {{
        "content": "Head is very painful recently"
    }},
    {{
        "content": "When did these headaches first start?",
        "label": "It also appeared a month ago"
    }},
    {{
        "content": "Have you had a cold or fever recently",
        "label": "Yes"
    }},
    {{
        "content": "Have you experienced any stiffness in your neck or difficulty moving it?",
        "label": "Around my left eye, Always on the same side."
    }},
    {{
        "content": "This could be viral meningitis caused by the cold. However, don't worry, it will likely improve soon. Still, I recommend you go to the hospital for a detailed examination to confirm.",
        "label": "From 15 minutes to 3 hours"
    }},
    {{
        "content": "Do you have any symptoms that occur at the same time as your headache?",
        "label": "Red eye"
    }},
    {{
        "content": "Is the headache: - Piercing or sharp, as in an electric shock feeling?",
        "label": "Yes"
    }},
    {{
        "content": "Cluster headache"
    }}
]
Based on the above requirements, examples, and the following information about the symptoms of the disease, you need to complete the following optimization of the diagnosis and treatment decision path, noting that the optimized path needs to be kept in English. 

The symptoms of the disease are:
{content}

The decision path is:
{route}

The optimized decision path is as follows:
"""

# 你是一个疾病名称提取器。下面的决策路径中可能包含有一个或多个疾病，你需要从下面的决策路径中提取出其中包含的疾病名称，并以数组格式返回。并且，需要优先抽取最终诊断结果中的疾病，如果决策路径中最后没有诊断出疾病，则返回空数组。
extract_disease_llm_prompt = '''You are a disease name extractor. The following decision path may contain one or more diseases. You need to extract the disease names contained in the following decision path and return them in array format. In addition, the diseases in the final diagnosis need to be extracted first, and if no diseases are diagnosed in the decision path, an empty array is returned.

Return only the extracted disease array, do not return anything that is not in array format.

Refer to the following a few examples:
## Example 1:
### Input:
[
    {{
        "content": "When did these headaches first start?Is this headache the same as ones you’ve had before or is it different in some way?",
        "label": "New headache (A headache of recent onset or a chronic headache that has changed in character.)"
    }},
    {{
        "content": "Tell me more about your headaches. Is there any othter symptoms?",
        "label": "Hemiparesis"
    }},
    {{
        "content": "Brain tumor, stroke, brain abscess"
    }}
]
### Output:
["Brain tumor", "stroke", "brain abscess"]

## Example 2:
### Input:
[
    {{
        "content": "How long has your diarrhea lasted?",
        "label": "Diarrhea lasting at least 4 weeks (Chronic )"
    }},
    {{
        "content": "Chronic diarrhea.Do you have any other symptoms?",
        "label": " abdominal bloating"
    }},
    {{
        "content": "Has there been passage of mucus?",
        "label": "yes"
    }},
    {{
        "content": "IBS"
    }}
]
### Output:
["IBS"]

## Example 3:
### Input:
[
    {{
        "content": "Do you have a history of Chronic liver disease or Peptic ulcer disease?",
        "label": "No"
    }},
    {{
        "content": "physical examination",
        "label": "Suggests lower respiratory tract source"
    }},
    {{
        "content": "Chest radiograph",
        "label": "Normal"
    }},
    {{
        "content": "Do you have a history of Chronic obstructive pulmonary disease?",
        "label": "Yes"
    }},
    {{
        "content": "Risk factors for Lung cancer"
    }}
]
### Output:
["Lung cancer"]

### Input:
{route}
### Output:
'''