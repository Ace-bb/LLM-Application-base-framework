# 对从App builder中取出来的应用进行改写
template = """你是一个Function接口优化器，你需要优化下面给出的函数内容，保证Function符合JSON schema规范。
优化的要求为：
1. 为Function定义一个英文名。
2. Function的description和参数的description都必须为中文介绍。
3. 必须符合JSON schema格式。
4. 优化Function的description部分，使函数介绍能够更加清晰的表达函数的作用和意图。
5. 优化Function参数的介绍，使每个参数的介绍更加清晰的表达参数的作用。
6. 优化后的结构必须按照原格式输出。

函数接口为：{description}

优化之后的结果为："""

# 对每个Function生成输入输出
qa_generate_template = """
# 角色
你是一个Function Calling任务数据生成器。下面给你一个Function信息，你需要根据Function内容，生成一个问题，这个问题需要调用给出的Function来进行回答，并且还需要根据问题生成调用函数Function的参数。

# 要求
1. 输出必须为JSON格式
2. 不能修改Function内容。
3. 生成的问题Question必须是能够调用Function解决，在Question中应该包含有需要在调用函数时需要传入的参数内容。
4. 生成的Answer必须是调用Function时需要传递的参数，并且参数内容可以全部从Question中获得。
5. Question和Answer中的参数值都需要以中文为主。
6. 最终输出的结果必须按照下面给出的输出格式输出，参考示例。

# 输出格式
{{

    "question": "",
    "function": [],
    "input_parameters": []
}}

# 示例
{{

    "question": "本科就读于浙江大学的简历",
    "function": [
        {{
            "name": "resume_search_by_conditions",
            "description": "能够根据给定的搜索条件去数据库检索出符合要求的全部简历，搜索条件的参数可以是一个或者多个，但是不能没有参数。",
            "parameters": {{
                "type": "dict",
                "properties": {{
                    "bkCollegeName": {{
                        "type": "array",
                        "items": {{
                            "type": "string"
                        }},
                        "description": "关键词列表；本科毕业学校的名称，对缩写、简写的名称进行扩充和联想，下面是部分缩写对应的学校名称：“清北”对应[\"清华大学\"、\"北京大学\"]，“浙大”对应[\"浙江大学\"]，“c9”代表的学校为：[\"北京大学\", \"清华大学\", \"复旦大学\", \"上海交通大学\", \"浙江大学\", \"南京大学\", \"中国科学技术大学\", \"哈尔滨工业大学\", \"西安交通大学\"]。查询中提到了多个则表示成数组，如果没有提到则不填。"
                    }},
                    "email": {{
                        "type": "string",
                        "description": "根据邮箱查询"
                    }},
                    "currentCities": {{
                        "type": "string",
                        "description": "现居住城市，如果没有提到则不填"
                    }}
                }},
                "required": []
            }}
        }}
    ],
    "input_parameters": [
        {{
            "function_name": "resume_search_by_conditions",
            "arguments": {{
                "bkCollegeName": ["浙江大学"]
            }}
        }}
    ]
}}

# 给定的Function
{function_intro}

# 输出：
"""

ICL_Generate_Prompt = """
# 角色
你是一个Function Calling任务数据生成器。下面给你一个Function信息，你需要根据Function内容，生成一个问题，这个问题需要调用给出的Function来进行回答，并且还需要根据问题生成调用函数Function的参数。

# 要求
1. 生成的Question为用户提出的问题，input_parameters是调用函数传入的参数
2. 生成的Question不能与下面已有的Question重复，并且语气、表达方式需要尽可能多样。
2. 不能修改Function内容。
3. 生成的问题Question必须是能够调用Function解决，在Question中应该包含有需要在调用函数时需要传入的参数内容。
4. 生成的Answer必须是调用Function时需要传递的参数，并且参数内容可以全部从Question中获得。
5. Question和input_parameters中的参数值都需要以中文为主。
6. 输出必须为JSON格式，最终输出的结果必须按照下面给出的输出格式输出，参考示例。

# 输出格式
{{

    "question": "",
    "function": [],
    "input_parameters": []
}}

# 已有Question
{questions}

# 示例
{{

    "question": "本科就读于浙江大学的简历",
    "function": [
        {{
            "name": "resume_search_by_conditions",
            "description": "能够根据给定的搜索条件去数据库检索出符合要求的全部简历，搜索条件的参数可以是一个或者多个，但是不能没有参数。",
            "parameters": {{
                "type": "dict",
                "properties": {{
                    "bkCollegeName": {{
                        "type": "array",
                        "items": {{
                            "type": "string"
                        }},
                        "description": "关键词列表；本科毕业学校的名称，对缩写、简写的名称进行扩充和联想，下面是部分缩写对应的学校名称：“清北”对应[\"清华大学\"、\"北京大学\"]，“浙大”对应[\"浙江大学\"]，“c9”代表的学校为：[\"北京大学\", \"清华大学\", \"复旦大学\", \"上海交通大学\", \"浙江大学\", \"南京大学\", \"中国科学技术大学\", \"哈尔滨工业大学\", \"西安交通大学\"]。查询中提到了多个则表示成数组，如果没有提到则不填。"
                    }},
                    "email": {{
                        "type": "string",
                        "description": "根据邮箱查询"
                    }},
                    "currentCities": {{
                        "type": "string",
                        "description": "现居住城市，如果没有提到则不填"
                    }}
                }},
                "required": []
            }}
        }}
    ],
    "input_parameters": [
        {{
            "function_name": "resume_search_by_conditions",
            "arguments": {{
                "bkCollegeName": ["浙江大学"]
            }}
        }}
    ]
}}

# 给定的Function
{function_intro}

# 输出：

"""

