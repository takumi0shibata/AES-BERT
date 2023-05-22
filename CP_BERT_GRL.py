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
from custom_layers.GRL import GradientReversalLayer

def main():
    parser = argparse.ArgumentParser(description='BERT_model')
    parser.add_argument('--prompt_id', type=int, default=1)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lambda_', type=float, default=0.1)
    
    args = parser.parse_args()

    data = pd.read_excel('data/training_set_rel3.xlsx')

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
    prompt_ids = [i for i in range(1, 9)]
    test_prompt_id = args.prompt_id
    train_prompt_id = [i for i in range(1, 9) if i != test_prompt_id]
    data_del = data.iloc[:, [0, 1, 2, 6]]
    data_del = data_del.dropna(subset=['domain1_score'])

    # 得点データを[0, 1]のスケールにする
    for id in prompt_ids:
        minscore, maxscore = asap_range[id][0], asap_range[id][1]
        data_del.loc[data_del['essay_set'] == id, 'normalized_score'] = (data_del.loc[data_del['essay_set'] == id, 'domain1_score'] - minscore) / (maxscore - minscore)

    # train と test にデータフレームを分割
    data_del_test = data_del[data_del['essay_set'] == test_prompt_id]
    data_del_train = data_del[data_del['essay_set'].isin(train_prompt_id)]

    print(f'Training model from prompt{train_prompt_id}')
    print(f'Test model on {test_prompt_id}')
    print(f'Number of essays for training: {len(data_del_train)}')
    print(f'Number of essays for test: {len(data_del_test)}')

    # データフレームの確認
    data_del.head()

    train_data, val_data = train_test_split(data_del_train, test_size=0.2, random_state=42)
    test_data = data_del_test

    # モデルの定義
    class TextRegressionModel_w_GRL(nn.Module):
        def __init__(self, model_name, num_labels):
            super(TextRegressionModel_w_GRL, self).__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.score_predictor = nn.Linear(self.bert.config.hidden_size, num_labels)
            self.domain_predictor0 = nn.Linear(self.bert.config.hidden_size, 100)
            self.domain_predictor1 = nn.Linear(100, 8)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.GRL = GradientReversalLayer(lambda_=args.lambda_)

        def forward(self, input_ids, attention_mask):
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            # 得点予測
            score = self.score_predictor(output.pooler_output)
            score = self.sigmoid(score)

            # ドメイン予測
            domain = self.GRL(output.pooler_output)
            domain = self.domain_predictor0(domain)
            domain = self.relu(domain)
            domain = self.domain_predictor1(domain)
            
            return score, domain

    # データセットの定義
    class TextRegressionDataset(Dataset):
        def __init__(self, texts, scores, prompt_id, tokenizer, max_length):
            self.texts = texts
            self.scores = scores
            self.prompt_id = prompt_id
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            score = self.scores[item]
            prompt_id = self.prompt_id[item]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'score': torch.tensor(score, dtype=torch.float),
                'prompt_id': torch.tensor(prompt_id, dtype=torch.int64)
            }

    # データローダーの定義
    def create_data_loader(df, tokenizer, max_length, batch_size):
        ds = TextRegressionDataset(
            texts=df.essay.to_numpy(),
            scores=df.normalized_score.to_numpy(),
            prompt_id=df.essay_set.to_numpy()-1,
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

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = TextRegressionModel_w_GRL(MODEL_NAME, num_labels=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps')
    print(device)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    #total_steps = len(train_data) * EPOCHS // BATCH_SIZE

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
    def train_epoch(model, data_loader, loss_fn0, loss_fn1, optimizer, device, scheduler):
        model.train()

        score_losses = []
        domain_losses = []
        progress_bar = tqdm(data_loader, desc="Training", unit="batch")
        for d in progress_bar:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['score'].to(device)
            domains = d['prompt_id'].to(device)

            score_outputs, domain_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            score_loss = loss_fn0(score_outputs.squeeze(), targets)
            domain_loss = loss_fn1(domain_outputs.squeeze(), domains)

            score_losses.append(score_loss.item())
            domain_losses.append(domain_loss.item())

            loss = score_loss + domain_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'score_loss': sum(score_losses) / len(score_losses),
                                      'domain_loss': sum(domain_losses) / len(domain_losses)})

        return np.mean(score_losses) + np.mean(domain_losses)

    # 評価関数の定義
    def evaluate_model(model, data_loader, loss_fn0, loss_fn1, device, asap_range, test_prompt_id):
        model.eval()

        score_losses = []
        domain_losses = []
        predictions = []
        targets = []

        progress_bar = tqdm(data_loader, desc="Evaluation", unit="batch")

        with torch.no_grad():
            for d in progress_bar:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                targets.extend(d['score'].tolist())

                score_outputs, domain_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                score_loss = loss_fn0(score_outputs.squeeze(), d['score'].to(device))
                domain_loss = loss_fn1(domain_outputs.squeeze(), d['prompt_id'].to(device))
                score_losses.append(score_loss.item())
                domain_losses.append(domain_loss.item())

                squeezed_outputs = score_outputs.squeeze()
                predictions.extend(squeezed_outputs.tolist() if squeezed_outputs.dim() > 0 else [squeezed_outputs.item()])
                
                progress_bar.set_postfix({'score_loss': sum(score_losses) / len(score_losses),
                                          'domain_loss': sum(domain_losses) / len(domain_losses)})

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        corr = np.corrcoef(targets, predictions)[0, 1]

        minscore, maxscore = asap_range[test_prompt_id][0], asap_range[test_prompt_id][1]
        rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(targets))
        rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(predictions))

        qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
        lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')

        return {
            'score_loss': np.mean(score_losses),
            'domain_loss': np.mean(domain_losses),
            'qwk': qwk,
            'lwk': lwk,
            'corr': corr,
            'rmse': rmse,
            'mae': mae
        }


    # 学習
    loss_fn_MSE = nn.MSELoss().to(device)
    loss_fn_CCE = nn.CrossEntropyLoss().to(device)
    best_test_metrics = [1, 1, 1, 1, 1]
    best_val_metrics = [1, 1, 1, 1, 1]
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn_MSE,
            loss_fn_CCE,
            optimizer,
            device,
            scheduler
        )

        print(f'Train loss: {train_loss}')

        val_history = evaluate_model(
            model,
            val_data_loader,
            loss_fn_MSE,
            loss_fn_CCE,
            device,
            asap_range,
            test_prompt_id
        )

        print(f'Validation loss: {val_history["score_loss"] + val_history["domain_loss"]}')

        # Evaluate the model on the test set
        test_history = evaluate_model(
            model,
            test_data_loader,
            loss_fn_MSE,
            loss_fn_CCE,
            device,
            asap_range,
            test_prompt_id
        )

        print(f'Test loss: {test_history["score_loss"] + test_history["domain_loss"]}')


        print(f'[VAL]  -> MAE: {val_history["mae"]}, CORR: {val_history["corr"]}, RMSE: {val_history["rmse"]}')
        print(f'[TEST] -> QWK: {test_history["qwk"]}, CORR: {test_history["corr"]}, RMSE: {test_history["rmse"]}')

        if val_history["rmse"] < best_val_metrics[-2]: #RMSEで評価
            for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                best_val_metrics[i] = val_history[met]
                best_test_metrics[i] = test_history[met]

        print(f'[BEST] -> QWK: {best_test_metrics[0]}, CORR: {best_test_metrics[2]}, RMSE: {best_test_metrics[3]}')

    import os
    output_dirs = './results/BERT_cross_prompt/'
    os.makedirs(output_dirs, exist_ok=True)
    pd.DataFrame(np.array(best_test_metrics).reshape(1, 5), columns=['qwk', 'lwk', 'corr', 'rmse', 'mae']).to_csv(output_dirs+'BERT_cross_prompt{}.csv'.format(test_prompt_id), index=False, header=True)

if __name__=='__main__':
    main()