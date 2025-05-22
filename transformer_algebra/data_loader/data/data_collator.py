from transformers import PreTrainedTokenizerFast as Tokenizer
from .preprocessor import AbstractPreprocessor
from typing import Dict
from torch.utils.data import Dataset


class StandardDataset(Dataset):
    def __init__(self, data_path: str, preprocessor: AbstractPreprocessor) -> None:
        self.data_path = data_path
        self.input_texts = []
        self.targets_texts = []
        self.preprocessor = preprocessor

        # Load and parse the data file
        with open(self.data_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue  # Skip empty lines

                # Split input and target expressions using "#" delimiter
                if "#" not in line:
                    continue  # Skip lines with unexpected format (no delimiter)

                input_part, target_part = line.split("#", 1)
                self.input_texts.append(input_part.strip())
                self.targets_texts.append(target_part.strip())

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get dataset item and convert to internal representation.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve

        Returns
        -------
        tuple
            A pair (src, tgt) of preprocessed source and target
        """
        src = self.preprocessor(self.input_texts[idx])
        tgt = self.preprocessor(self.targets_texts[idx])
        return {"input": src, "target": tgt}

    def __len__(self) -> int:
        return len(self.input_texts)

    
class StandardDataCollator:
    def __init__(self, tokenizer: Tokenizer = None) -> None:
        self.tokenizer = tokenizer
    
    def _pad_sequences(self, sequences, padding_value=0):
        """Pads a list of sequences and converts them to a tensor."""
        # Calculate the maximum length of the sequences.
        max_length = max(len(seq) for seq in sequences)
        
        # Apply padding.
        padded_sequences = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            # Pad the sequence with the specified padding value.
            padded_seq = seq + [padding_value] * padding_length
            padded_sequences.append(padded_seq)
        
        # '+2' for bos/eos tokens.
        # Initialize a tensor of zeros with the appropriate shape.
        padded = torch.zeros(len(sequences), max_length + 2, dtype=torch.long)
        # Fill the tensor with the padded sequences, leaving space for BOS/EOS tokens.
        padded[:, 1:max_length+1] = torch.tensor(padded_sequences)
        
        return padded
    
    def __call__(self, batch):
        """
        Collates a batch of data samples.
        If a tokenizer is provided, it tokenizes 'input' and 'target' attributes.
        Other attributes starting with 'target_' are prefixed with 'decoder_' and padded.
        """
        batch_dict = {}
        
        # Get the attributes from the first item in the batch.
        attributes = batch[0].keys()
        
        if self.tokenizer is None:
            # If no tokenizer is provided, return the batch as is.
            for attribute in attributes:
                attribute_batch = [item[attribute] for item in batch]
                batch_dict[attribute] = attribute_batch

            return batch_dict
            
        for attribute in attributes: 
            attribute_batch = [item[attribute] for item in batch]
            
            if attribute == 'input':
                # Tokenize the input sequences.
                inputs = self.tokenizer(attribute_batch, padding='longest', return_tensors='pt')
                batch_dict['input_ids'] = inputs['input_ids']
                batch_dict['attention_mask'] = inputs['attention_mask']
                
            elif attribute == 'target':
                # Tokenize the target sequences.
                targets = self.tokenizer(attribute_batch, padding='longest', return_tensors='pt')
                # Prepare decoder input ids (remove the last token, usually EOS).
                batch_dict['decoder_input_ids'] = targets['input_ids'][:, :-1].contiguous()
                # Prepare decoder attention mask accordingly.
                batch_dict['decoder_attention_mask'] = targets['attention_mask'][:, :-1].contiguous()
                
                # Prepare labels for the loss calculation (shift by one, usually remove BOS).
                labels = targets['input_ids'][:, 1:].contiguous()
                label_attention_mask = targets['attention_mask'][:, 1:].contiguous().bool()
                # Set padding tokens in labels to -100 to be ignored by the loss function.
                labels[~label_attention_mask] = -100
                batch_dict['labels'] = labels
                
            else:
                # For other attributes, if they start with 'target_',
                # prefix them with 'decoder_' (e.g., 'target_aux' becomes 'decoder_aux').
                if attribute.startswith('target_'):
                    attribute_key = 'decoder_' + attribute[7:] #  Corrected key for batch_dict
                else:
                    attribute_key = attribute # Use original attribute name if no prefix
                # Pad the sequences for these attributes.
                batch_dict[attribute_key] = self._pad_sequences(attribute_batch, padding_value=0)
                
        return batch_dict
