from LLM.llm import LLM

from conf.Tools import Tools
tools = Tools()
import json, sys
from tqdm import tqdm
import traceback

def format_checker(function, input_parameters):
    try:
        if function["name"] != input_parameters["function_name"]: return False
        required_properties = function["parameters"]["required"]
        for k in required_properties:
            if k not in input_parameters["arguments"].keys(): continue
        
        llm = LLM()
        llm.setTemplate(temp="""下面给你一个函数的描述，其中包括需要传入的参数信息。你需要判断给定的参数是否符合这个函数的参数要求，符合返回True，不符合返回False。

# 函数描述
{function}

# 给定的参数
{para}

# 给定的参数是否符合函数描述中的参数要求：""", values = ["function", "para"])
        llm.run({"function": function, "para": input_parameters})
        return True
    except:
        return False


def generate_function_qa(_id, item):
    llm = LLM()
    llm.setTemplate(temp=ICL_Generate_Prompt, values=["questions", "function_intro"])
    cur_function_qas = list()
    cur_function_qas.append(item)
    # wf = open(f"datas/all_function_res_qas/{_id}.jsonl", 'a+', encoding="utf-8")
    # wf.write("fffffffffffffffffffffffffffffffffffffff\n")
    for i in tqdm(range(30), desc=str(_id)):
        tn = 5
        while tn>0:
            try:
                # print([f"{qid}. {q['question']}" for qid,q in enumerate(cur_function_qas)])
                res = llm.run({"function_intro": item["function"], "questions": "\n".join([f"{qid}. {q['question']}" for qid,q in enumerate(cur_function_qas)])})
                ge_f = json.loads(res.replace('```json', '').replace('```', ''))
                if format_checker(ge_f["function"][0], ge_f["input_parameters"][0]):
                    cur_function_qas.append(ge_f)
                    wf = open(f"datas/all_function_res_qas/{_id}.jsonl", 'a+', encoding="utf-8")
                    wf.write(json.dumps(ge_f, ensure_ascii=False)+ "\n")
                    wf.close()
                    break
            except Exception as e:
                print(f"line: {sys._getframe().f_lineno}\t {e} \ntraceback.print_exc(): {traceback.print_exc()}")
                tn-=1
                continue
    # wf.close()
    return cur_function_qas

from conf.prompt import ICL_Generate_Prompt
import json, sys
qa_function_data = tools.read_json("/root/Projects/LLM-Application-base-framework/datas/genereated_qa_json1.json")

llm = LLM()
llm.setTemplate(temp=ICL_Generate_Prompt, values=["questions", "function_intro"])
all_function_res_qas = list()
run_paras = list()
for _id, item in enumerate(qa_function_data[:]):
    if _id<12: continue
    run_paras.append((_id, item))
    continue
    cur_function_qas = list()
    cur_function_qas.append(item)

    gn = 5
    while gn>0:
        print(gn)
        # try:
        res = llm.run({"function_intro": item["function"], "questions": "\n".join([f"{qid}. {q['question']}" for qid,q in enumerate(cur_function_qas)])})
        ge_f = json.loads(res.replace('```json', '').replace('```', ''))
        if format_checker(ge_f["function"][0], ge_f["input_parameters"][0]):
            cur_function_qas.append(ge_f)
            gn-=1
        # except Exception as e:
        #     print(f"line: {sys._getframe().f_lineno}\t {e}")
        #     print(res)
        #     continue
    
    all_function_res_qas.append(cur_function_qas)

all_function_res_qas = tools.multi_thread_run(6, generate_function_qa, run_paras, "Generate")
tools.write_2_json(all_function_res_qas, f"./datas/all_function_res_qas.json")
