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
from utils.utils import get_min_max_scores

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

def main():
    parser = argparse.ArgumentParser(description='BERT_model')
    parser.add_argument('--prompt_id', type=int, default=1)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    #データの読み込み
    test_prompt_id = args.prompt_id
    data = pd.read_csv(f'ASAPplus/data{test_prompt_id}.csv')
    data = data.dropna(subset=['Overall'])

    # 得点データを[0, 1]のスケールにする
    for key, value in get_min_max_scores()[test_prompt_id].items():
        minscore, maxscore = value
        data.loc[:, key] = (data[key] - minscore) / (maxscore - minscore)
    

    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # データローダーの定義
    def create_data_loader(df, tokenizer, max_length, batch_size):
        ds = TextRegressionDataset(
            texts=df.essay.to_numpy(),
            scores=df[get_min_max_scores()[test_prompt_id].keys()].to_numpy(),
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
    model = TextRegressionModel(MODEL_NAME, num_labels=len(get_min_max_scores()[test_prompt_id]))

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
            loss = loss_fn(outputs, targets)

            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

        return np.mean(losses)

    # 評価関数の定義
    def evaluate_model(model, data_loader, loss_fn, device, attribute_dict, test_prompt_id):
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
                loss = loss_fn(outputs, d['score'].to(device))
                losses.append(loss.item())

                predictions.extend(outputs.tolist())
                
                progress_bar.set_postfix({'loss': sum(losses) / len(losses)})

        qwk_list, lwk_list, corr_list, rmse_list, mae_list = [], [], [], [], []
        targets = np.array(targets)
        predictions = np.array(predictions)
        print(targets.shape)
        print(predictions.shape)
        for i, key in enumerate(attribute_dict.keys()):
            rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            corr = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
            rmse_list.append(rmse)
            mae_list.append(mae)
            corr_list.append(corr)

            minscore, maxscore = attribute_dict[key]
            rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(targets[:, i]))
            rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(predictions[:, i]))
            qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
            lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')
            qwk_list.append(qwk)
            lwk_list.append(lwk)

        return {
            'loss': np.mean(losses),
            'qwk': np.array(qwk_list),
            'lwk': np.array(lwk_list),
            'corr': np.array(corr_list),
            'rmse': np.array(rmse_list),
            'mae': np.array(mae_list)
        }

    class MultiOutputMSELoss(nn.Module):
            def __init__(self):
                super(MultiOutputMSELoss, self).__init__()
                self.mse_loss = nn.MSELoss()

            def forward(self, outputs, targets):
                # Assuming outputs and targets are of shape [batch_size, num_outputs]
                total_loss = 0
                for i in range(outputs.shape[1]):
                    total_loss += self.mse_loss(outputs[:, i], targets[:, i])
                return total_loss / outputs.shape[1] # return the average loss


    # 学習
    loss_fn = MultiOutputMSELoss().to(device)
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
            get_min_max_scores()[test_prompt_id],
            test_prompt_id
        )

        print(f'Validation loss: {val_history["loss"]}')

        # Evaluate the model on the test set
        test_history = evaluate_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            get_min_max_scores()[test_prompt_id],
            test_prompt_id
        )

        print(f'Test loss: {test_history["loss"]}')


        print(f'[VAL]  -> QWK: {np.mean(val_history["qwk"])}, CORR: {np.mean(val_history["corr"])}, RMSE: {np.mean(val_history["rmse"])}')
        print(f'[TEST] -> QWK: {np.mean(test_history["qwk"])}, CORR: {np.mean(test_history["corr"])}, RMSE: {np.mean(test_history["rmse"])}')

        if np.mean(val_history["qwk"]) > best_val_metrics[0]:
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = np.mean(val_history[met])
                best_test_metrics[i] = np.mean(test_history[met])
                best_qwk = test_history["qwk"]
                best_lwk = test_history["lwk"]
                best_corr = test_history["corr"]
                best_rmse = test_history["rmse"]
                best_mae = test_history["mae"]

            import os
            output_dir = 'results/BERT_PS_Analytic/'
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), output_dir+f'model_state_dict{test_prompt_id}.pth')
                

        print(f'[BEST] -> QWK: {best_test_metrics[0]}, CORR: {best_test_metrics[2]}, RMSE: {best_test_metrics[3]}')
        for i, key in enumerate(get_min_max_scores()[test_prompt_id].keys()):
            print(f'{key}: {best_qwk[i]}')

    df = pd.DataFrame(np.array([best_qwk.tolist(), best_lwk.tolist(), best_corr.tolist(), best_rmse.tolist(), best_mae.tolist()]),
                      index=['qwk', 'lwk', 'corr', 'rmse', 'mae'], columns=get_min_max_scores()[test_prompt_id].keys())
    df.to_csv(output_dir+f'prompt{test_prompt_id}.csv')

if __name__=='__main__':
    main()