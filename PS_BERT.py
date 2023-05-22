import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error
from tqdm import tqdm
import argparse

from utils.dataset import TextRegressionDataset

def main():
    parser = argparse.ArgumentParser(description='BERT_model')
    parser.add_argument('--prompt_id', type=int, default=1)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    data = pd.read_excel('data/training_set_rel3.xlsx')

    # Show Data
    data.head()

    asap_range = {
        1: [2, 12],
        2: [1, 6],
        3: [0, 3],
        4: [0, 3],
        5: [0, 4],
        6: [0, 4],
        7: [0, 30],
        8: [0, 60]
    }

    # データを必要な部分だけ切り取り
    test_prompt_id = args.prompt_id
    data_del = data.iloc[:, [0, 1, 2, 6]]
    data_del = data_del[data_del['essay_set']==test_prompt_id]
    data_del = data_del.dropna(subset=['domain1_score'])

    # 得点データを[0, 1]のスケールにする
    minscore, maxscore = asap_range[test_prompt_id][0], asap_range[test_prompt_id][1]
    data_del['normalized_score'] = (data_del['domain1_score'] - minscore) / (maxscore - minscore)

    # データフレームの確認
    data_del.head()

    train_data, temp_data = train_test_split(data_del, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # モデルの定義
    class TextRegressionModel(nn.Module):
        def __init__(self, model_name, num_labels):
            super(TextRegressionModel, self).__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.out(output.pooler_output)
            logits = self.sigmoid(logits)
            
            return logits

    # データローダーの定義
    def create_data_loader(df, tokenizer, max_length, batch_size):
        ds = TextRegressionDataset(
            texts=df.essay.to_numpy(),
            scores=df.normalized_score.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_length
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0
        )

    # Set up hyperparameters
    MODEL_NAME = args.model
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TextRegressionModel(MODEL_NAME, num_labels=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps')
    print(device)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data) * EPOCHS // BATCH_SIZE

    # データローダーのセットアップ
    train_data_loader = create_data_loader(train_data, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_data_loader = create_data_loader(val_data, tokenizer, MAX_LENGTH, BATCH_SIZE)
    test_data_loader = create_data_loader(test_data, tokenizer, MAX_LENGTH, BATCH_SIZE)

    num_training_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 訓練関数の定義
    def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
        model.train()

        losses = []
        progress_bar = tqdm(data_loader, desc="Training", unit="batch")
        for d in progress_bar:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['score'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs.squeeze(), targets)

            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

        return np.mean(losses)

    # 評価関数の定義
    def evaluate_model(model, data_loader, loss_fn, device, asap_range, test_prompt_id):
        model.eval()

        losses = []
        predictions = []
        targets = []

        progress_bar = tqdm(data_loader, desc="Evaluation", unit="batch")

        with torch.no_grad():
            for d in progress_bar:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                targets.extend(d['score'].tolist())

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), d['score'].to(device))
                losses.append(loss.item())

                squeezed_outputs = outputs.squeeze()
                predictions.extend(squeezed_outputs.tolist() if squeezed_outputs.dim() > 0 else [squeezed_outputs.item()])
                
                progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        corr = np.corrcoef(targets, predictions)[0, 1]

        minscore, maxscore = asap_range[test_prompt_id][0], asap_range[test_prompt_id][1]
        rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(targets))
        rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(predictions))

        qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
        lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')

        return {
            'loss': np.mean(losses),
            'qwk': qwk,
            'lwk': lwk,
            'corr': corr,
            'rmse': rmse,
            'mae': mae
        }

    # 学習
    loss_fn = nn.MSELoss().to(device)
    best_test_metrics = [-1, -1, -1, -1, -1]
    best_val_metrics = [-1, -1, -1, -1, -1]
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_data)
        )

        print(f'Train loss: {train_loss}')

        val_history = evaluate_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            asap_range,
            test_prompt_id
        )

        print(f'Validation loss: {val_history["loss"]}')

        # Evaluate the model on the test set
        test_history = evaluate_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            asap_range,
            test_prompt_id
        )

        print(f'Test loss: {test_history["loss"]}')


        print(f'[VAL]  -> QWK: {val_history["qwk"]}, CORR: {val_history["corr"]}, RMSE: {val_history["rmse"]}')
        print(f'[TEST] -> QWK: {test_history["qwk"]}, CORR: {test_history["corr"]}, RMSE: {test_history["rmse"]}')

        if val_history["qwk"] > best_val_metrics[0]:
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = val_history[met]
                best_test_metrics[i] = test_history[met]

        print(f'[BEST] -> QWK: {best_test_metrics[0]}, CORR: {best_test_metrics[2]}, RMSE: {best_test_metrics[3]}')

    pd.DataFrame(np.array(best_test_metrics).reshape(1, 5), columns=['qwk', 'lwk', 'corr', 'rmse', 'mae']).to_csv('results/BERT_prompt_specific/BERT_prompt_specific{}.csv'.format(test_prompt_id), index=False, header=True)

if __name__=='__main__':
    main()