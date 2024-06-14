# Seq2Seq Model for German to English Translation

This repository contains the implementation of a sequence-to-sequence (Seq2Seq) model for translating German sentences into English using PyTorch. The model utilizes an encoder-decoder architecture and is trained and evaluated on a dataset of German-English sentence pairs.

## Table of Contents

- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.x
- PyTorch
- TorchText
- TensorBoard

Install the required packages via pip:

```bash
pip install torch torchtext tensorboard
```

## Model Architecture

### Encoder

The encoder processes the input German sentence and produces a context vector that summarizes the input sequence.

- **Embedding Layer**: Converts input tokens into dense vectors.
- **LSTM Layer**: Processes the embedded tokens to produce hidden and cell states.

### Decoder

The decoder generates the output English sentence based on the context vector provided by the encoder.

- **Embedding Layer**: Converts output tokens into dense vectors.
- **LSTM Layer**: Uses the context vector and previous hidden states to generate the next token.
- **Linear Layer**: Maps the LSTM output to the vocabulary space.

## Training

### Hyperparameters

- `num_epochs = 10`
- `learning_rate = 0.001`
- `batch_size = 64`
- `encoder_embedding_size = 512`
- `decoder_embedding_size = 512`
- `hidden_size = 1024`
- `num_layers = 2`
- `enc_dropout = 0.1`
- `dec_dropout = 0.1`

### Optimizer and Loss Function

- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss with padding token ignored

### Data Iterators

Data is loaded using TorchText's `BucketIterator`, which ensures efficient batching and padding.

## Evaluation

### Translation Function

The `translate_sentence` function translates a given German sentence into English using the trained model.

```python
def translate_sentence(model, sentence, german, english, device, max_length=50):
    model.eval()
    tokens = german.tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    sentence_tensor = torch.tensor([german.vocab.stoi[token] for token in tokens]).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]
    for _ in range(max_length):
        previous_word = torch.tensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return ' '.join(translated_sentence)
```

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/seq2seq-translation.git
    cd seq2seq-translation
    ```

2. **Prepare the data**:
    Ensure you have the German-English dataset. Use TorchText to load and preprocess the data.

3. **Train the model**:
    ```python
    # Initialize model, optimizer, and loss function
    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch + 1} / {num_epochs}]")

        model.eval()
        translated_sentence = translate_sentence(
            model, sentence, german, english, device, max_length=50
        )
        print(f"Translated example sentence: \n {translated_sentence}")
        model.train()

        epoch_loss = 0
        for batch_idx, batch in enumerate(train_iterator):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)
            optimizer.zero_grad()
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

        print(f"Epoch Loss: {epoch_loss}")
        losses.append(epoch_loss)

    # Evaluate the model
    score = bleu(test_data[1:100], model, german, english, device)
    print(f"Bleu score {score*100:.2f}")
    ```

4. **Translate sentences**:
    ```python
    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."
    translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=50)
    print(f"Translated sentence: {translated_sentence}")
    ```

## Acknowledgements

This implementation is inspired by various Seq2Seq models and tutorials available online. Special thanks to the PyTorch and TorchText communities for their comprehensive documentation and support.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
