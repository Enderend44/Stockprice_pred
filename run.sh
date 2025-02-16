#!/bin/bash

# Répertoire des logs
LOG_DIR="training_logs"
mkdir -p "$LOG_DIR"

# Modèles disponibles : transformer ou lstm
MODELS=("transformer" "lstm")

# Hyperparamètres à tester
BATCH_SIZES=(1024)
SEQ_LENGTHS=(50 100 200)
EPOCHS_LIST=(100)
LEARNING_RATES=(0.001 0.0005 0.0001)

# Boucle sur les hyperparamètres
for MODEL in "${MODELS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
            for EPOCHS in "${EPOCHS_LIST[@]}"; do
                for LR in "${LEARNING_RATES[@]}"; do
                    # Génération d'un timestamp pour identifier chaque run
                    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

                    # Log du run actuel
                    echo "Lancement de l'entraînement :"
                    echo "  Modèle       : $MODEL"
                    echo "  Batch size   : $BATCH_SIZE"
                    echo "  Sequence len : $SEQ_LENGTH"
                    echo "  Epochs       : $EPOCHS"
                    echo "  Learning rate: $LR"

                    # Commande d'entraînement
                    python3 train.py \
                        --model $MODEL \
                        --batch_size $BATCH_SIZE \
                        --seq_length $SEQ_LENGTH \
                        --epochs $EPOCHS \
                        --lr $LR \
                        > "$LOG_DIR/${MODEL}_${TIMESTAMP}.log" 2>&1

                    echo "Entraînement terminé pour ce jeu d'hyperparamètres. Les logs sont sauvegardés dans $LOG_DIR/${MODEL}_${TIMESTAMP}.log"
                done
            done
        done
    done
done

echo "Toutes les combinaisons d'hyperparamètres ont été exécutées."
