python script/eval_latex_table_correct_rate.py \
    --api_url http://10.70.0.128:30303 \
    --inference_result_folder qwen_html \
    --max_tokens 4096 \
    --system_prompt "" \
    --prompt "Please describe the picture using HTML while considering merged columns and merged row cells." \
    --target_platform windows \
    --datasets \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/train-data/20250211-人工-合併表格+合併儲存格/20250211-test/train-data分割出來的/orig:detect_table=False \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/train-data/20250211-人工-合併表格+合併儲存格/20250211-test/20250206-晚標記完/orig:detect_table=False \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/train-data/20250211-人工-合併表格+合併儲存格/20250211-test/第四次-單表格/orig \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/train-data/20250211-人工-合併表格+合併儲存格/20250211-test/第十次-單表格/orig \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/train-data/20250211-人工-合併表格+合併儲存格/20250211-test/第十二次-單表格/orig \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/train-data/20250211-人工-合併表格+合併儲存格/20250211-test/第十三次-單表格/orig
        