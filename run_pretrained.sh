#!/bin/bash

# 创建输出目录
mkdir -p output

# 定义要测试的模型列表
MODELS=("resnet18" "resnet34" "resnet50" "resnet101" "resnext50" "wide_resnet50" "densenet" "efficientnet")

# 定义优化器和学习率调度器
OPTIMIZERS=("adam" "adamw")
SCHEDULERS=("cosine" "onecycle")

# 设置最大字符数
MAX_CHARS=6

# 如果传入了参数，则只运行特定的模型
if [ $# -eq 1 ]; then
    if [[ " ${MODELS[@]} " =~ " $1 " ]]; then
        MODELS=("$1")
        echo "将只使用 $1 模型进行训练"
    else
        echo "未知的模型: $1"
        echo "可用模型: ${MODELS[@]}"
        exit 1
    fi
fi

# 测试导入PyTorch
python -c "import torch; print('PyTorch可用，版本:', torch.__version__)"
if [ $? -ne 0 ]; then
    echo "PyTorch导入失败，请检查环境配置"
    exit 1
fi

# 测试目录结构
if [ ! -d "code/train" ]; then
    echo "错误: 训练代码目录 'code/train' 不存在"
    echo "当前目录结构:"
    ls -la
    exit 1
fi

# 保存初始工作目录
SCRIPT_DIR=$(pwd)

# 对每个模型运行训练和测试
for MODEL in "${MODELS[@]}"; do
    # 对每个优化器运行
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        # 对每个学习率调度器运行
        for SCHEDULER in "${SCHEDULERS[@]}"; do
            # 创建模型特定的输出目录
            MODEL_DIR="output/${MODEL}_${OPTIMIZER}_${SCHEDULER}"
            mkdir -p "$MODEL_DIR"
            
            echo "========================================================"
            echo "开始训练模型: $MODEL, 优化器: $OPTIMIZER, 调度器: $SCHEDULER"
            echo "最大字符数: $MAX_CHARS"
            echo "========================================================"
            
            # 恢复到初始目录
            cd "$SCRIPT_DIR" || { echo "无法切换到初始目录"; exit 1; }
            
            # 检查训练代码目录是否存在
            if [ ! -d "code/train" ]; then
                echo "错误: 训练代码目录 'code/train' 不存在"
                continue
            fi
            
            # 运行训练
            cd code/train || { echo "无法切换到训练目录 code/train"; continue; }
            python train.py \
                --data_dir ../../tcdata/images/extracted \
                --json_dir ../../tcdata/json \
                --batch_size 500 \
                --epochs 100 \
                --lr 0.001 \
                --output_dir "../../$MODEL_DIR" \
                --backbone "$MODEL" \
                --optimizer "$OPTIMIZER" \
                --scheduler "$SCHEDULER" \
                --weight_decay 1e-5 \
                --max_chars $MAX_CHARS
            
            # 检查训练是否成功
            if [ $? -ne 0 ]; then
                echo "训练失败，跳过预测步骤"
                cd "$SCRIPT_DIR" || { echo "无法切换到初始目录"; exit 1; }
                continue
            fi
            
            # 恢复到初始目录
            cd "$SCRIPT_DIR" || { echo "无法切换到初始目录"; exit 1; }
            
            # 检查测试代码目录是否存在
            if [ ! -d "code/test" ]; then
                echo "错误: 测试代码目录 'code/test' 不存在"
                continue
            fi
            
            # 运行预测
            cd code/test || { echo "无法切换到测试目录 code/test"; continue; }
            python inference.py \
                --test_dir ../../tcdata/images/extracted/mchar_test_a \
                --model_path "../../$MODEL_DIR/best_model.pth" \
                --config_path "../../$MODEL_DIR/config.txt" \
                --output_file "../../$MODEL_DIR/submit.csv" \
                --backbone "$MODEL" \
                --max_chars $MAX_CHARS
            
            # 检查预测是否成功
            if [ $? -ne 0 ]; then
                echo "预测失败"
                cd "$SCRIPT_DIR" || { echo "无法切换到初始目录"; exit 1; }
                continue
            fi
            
            echo "模型 $MODEL 使用 $OPTIMIZER 优化器和 $SCHEDULER 调度器的训练与预测完成"
            echo "结果保存在 $MODEL_DIR 目录下"
            echo ""
            
            # 返回项目根目录
            cd "$SCRIPT_DIR" || { echo "无法切换到初始目录"; exit 1; }
        done
    done
done

# 汇总结果
echo "所有模型训练与预测完成，汇总结果..."

# 创建汇总结果的表格
RESULT_FILE="output/results_summary.txt"
echo "模型 | 优化器 | 调度器 | 验证集准确率 | 推理时间(ms/图) | 最大字符数" > "$RESULT_FILE"
echo "--- | --- | --- | --- | --- | ---" >> "$RESULT_FILE"

# 提取每个模型组合的结果
for MODEL in "${MODELS[@]}"; do
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        for SCHEDULER in "${SCHEDULERS[@]}"; do
            MODEL_DIR="output/${MODEL}_${OPTIMIZER}_${SCHEDULER}"
            
            # 提取验证集准确率
            if [ -f "$MODEL_DIR/training_log.txt" ]; then
                # 获取最后一行，提取序列准确率
                ACCURACY=$(tail -n 1 "$MODEL_DIR/training_log.txt" | cut -d',' -f5)
                
                # 提取推理时间
                if [ -f "$MODEL_DIR/inference_stats.json" ]; then
                    # 使用grep和sed提取推理时间
                    INFERENCE_TIME=$(grep "avg_time_per_image_ms" "$MODEL_DIR/inference_stats.json" | sed 's/.*: \(.*\),/\1/')
                    MAX_CHARS_USED=$(grep "max_chars" "$MODEL_DIR/inference_stats.json" | sed 's/.*: \(.*\),\?/\1/')
                    
                    # 添加到结果表
                    echo "$MODEL | $OPTIMIZER | $SCHEDULER | $ACCURACY | $INFERENCE_TIME | $MAX_CHARS_USED" >> "$RESULT_FILE"
                else
                    echo "$MODEL | $OPTIMIZER | $SCHEDULER | $ACCURACY | N/A | $MAX_CHARS" >> "$RESULT_FILE"
                fi
            else
                echo "$MODEL | $OPTIMIZER | $SCHEDULER | N/A | N/A | $MAX_CHARS" >> "$RESULT_FILE"
            fi
        done
    done
done

echo "结果汇总完成，详见 $RESULT_FILE"
echo "完成！" 