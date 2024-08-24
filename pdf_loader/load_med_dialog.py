import os
import json
from tqdm import tqdm


def write2jsonl(write_path, json_data):
    print(len(json_data))
    try:
        with open(write_path, 'a', encoding="utf-8") as f:
            for i in tqdm(range(len(json_data)), desc=f'Write to jsonl {write_path.split("/")[-1]}'):
                f.write(json.dumps(json_data[i], ensure_ascii=False)+"\n")
        return True
    except:
        return False
        
        
def load_med_dialog_to_json():
    # MedDialog_path = "/media/PJLAB\libinbin/Elements/Projects/DecisionTreeSearch/DialogDatasets/MedDialog/Medical-Dialogue-Dataset-Chinese"
    # MedDialog_json_sae_path = "/media/PJLAB\libinbin/Elements/Projects/DecisionTreeSearch/DialogDatasets/MedDialog/MedDialogueDataset"
    # jsonl_sae_path = "/media/PJLAB\libinbin/Elements/Projects/DecisionTreeSearch/DialogDatasets/MedDialog/MedDialogueDataset_jsonl"
    MedDialog_path = "/root/nas/projects/Datasets/MedDialogue/Medical-Dialogue-Dataset-Chinese"
    MedDialog_json_sae_path = "/root/nas/projects/Datasets/MedDialogue/MedDialogueDataset"
    jsonl_sae_path = "/root/nas/projects/Datasets/MedDialogue/MedDialogueDataset_jsonl"
    med_files = os.listdir(MedDialog_path)
    dataset_info = {}
    for med_f in med_files:
        med_file_data = list()
        with open(f"{MedDialog_path}/{med_f}", "r", encoding="utf-8") as f:
            med_dialogs = f.readlines()
            dialog_item = {}
            current_key = ""
            for row in tqdm(med_dialogs, total=len(med_dialogs), desc=f"Load {med_f}"):
                if row=="" or row=="\n" or len(row)==0 : continue # or row.startswith("https")
                # row = json.loads(line)
                if row.startswith("id"):
                    if dialog_item!={}:
                        med_file_data.append(dialog_item)
                        dialog_item={}
                    dialog_item["id"] = row.split("=")[-1]
                    dialog_item["urls"] = ""
                    current_key = "urls"
                    # print(f'ffff:{len(f)} file:{med_f}--id:{row.split("=")[-1]}')
                    
                elif row.startswith("Doctor faculty"):
                    dialog_item["Doctor_faculty"] = ""
                    current_key = "Doctor_faculty"
                elif row.startswith("Description"):
                    dialog_item["Description"] = ""
                    current_key = "Description"
                elif row.startswith("Dialogue"):
                    dialog_item["Dialogue"] = ""
                    current_key = "Dialogue"
                else:
                    if dialog_item[current_key]!="" or len(dialog_item[current_key])!=0 or not dialog_item[current_key].endswith("\n"):
                        dialog_item[current_key] += "\n"
                    dialog_item[current_key] += row.replace("\n", "")
        
        dataset_info[med_f] = {
            "total_num": len(med_file_data),
            "save_path": f'{MedDialog_json_sae_path}/{med_f.replace(".txt", ".json")}'
        }
        for i in tqdm(range(len(med_file_data)), desc="parse_disease"):
            if 'Description' not in med_file_data[i].keys(): continue
            Description = med_file_data[i]['Description']
            disease = ""
            detail = ""
            past_treatment = ""
            purpose = ""
            if "想得到怎样的帮助" in Description:
                purpose = Description.split("想得到怎样的帮助")[-1].strip(":").strip("：")
                Description = Description.split("想得到怎样的帮助")[0]
            if "曾经治疗情况和效果" in Description:
                past_treatment = Description.split("曾经治疗情况和效果")[-1].strip(":").strip("：")
                Description = Description.split("曾经治疗情况和效果")[0]
            if "病情描述（发病时间、主要症状、就诊医院等）" in Description:
                detail = Description.split("病情描述（发病时间、主要症状、就诊医院等）")[-1].strip(":").strip("：")
                Description = Description.split("病情描述（发病时间、主要症状、就诊医院等）")[0]
            if "疾病：" in Description:
                disease = Description.split("疾病：")[-1].strip(":").strip("：")
                Description = Description.split("疾病：")[0]
                
            med_file_data[i]['Description'] = {
                "disease": disease.replace("\n", ""),
                "detail": detail.replace("\n", ""),
                "past_treatment": past_treatment.replace("\n", ""),
                "purpose": purpose.replace("\n", "")
            }
            
        with open(f'{MedDialog_json_sae_path}/{med_f.replace(".txt", ".json")}', "w", encoding="utf-8") as f:
            json.dump(med_file_data, f, ensure_ascii=False)
        write2jsonl(f"{jsonl_sae_path}/{med_f.replace('.txt', '.jsonl')}", med_file_data)

def rewrite_jsonl():
    MedDialog_json_sae_path = "/root/nas/projects/Datasets/MedDialogue/MedDialogueDataset"
    jsonl_sae_path = "/root/nas/projects/Datasets/MedDialogue/MedDialogueDataset_jsonl"
    
    med_files = os.listdir(MedDialog_json_sae_path)
    dataset_info = {}
    for med_f in med_files:
        print(med_f)
        with open(f"{MedDialog_json_sae_path}/{med_f}", 'r', encoding="utf-8") as f:
            write2jsonl(f"{jsonl_sae_path}/{med_f}l", json.load(f))

# def 
if __name__=="__main__":
    # load_med_dialog_to_json()
    rewrite_jsonl()