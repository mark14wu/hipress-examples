horovodrun -np 4 -H local-docker:4 python3 finetune_classifier.py --task_name CoLA --lr 2e-5 --batch_size 32 --seed 7800 --epochs 10
