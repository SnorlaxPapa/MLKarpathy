# An explanation for multi-headed attention

---

## Keys, queries, and values
* During a forward pass, a key, query, and value matrix are generated for each character with respect to their current context in the form of embedding * Wk or Wq or Wv respectively. 
    * **Value:** The value of the character that is passed forward through the model
    * **Query:** The query is used to 'query' its relation to other characters in other positions. 
    * **Key:** A character's key is used to match to another character's query. Think of it as a response to the query.

---

# In each attention head:
* A dot product is found between the query and the key, the result of which constitutes a part of the attention mechanism (i.e. how much attention should my current character pay attention to this character). Recall how a dot product is a determinant for the similarity between two embeddings. In this case, the matmul of Q @ K.T results in the dot product between the two matrices, where Q and K are the query and key matrices of every character stacked together. *Note:* Attention values for future characters are masked. (For decoders, encoders will take the whole context)
* This dot product is scaled by 1/sqrt(dk), where dk is the head size (output dimensions of a single head). Why do we scale by this specific factor? When we do Q @ K.T, n = head_size operations are performed for each resulting element. This will lead to an increased variance equivalent to that of the head size (which isn't great for the softmax down the road). We scale down by sqrt(dk) so as to normalize this distribution and then softmax it. This is our attention factor.
* The attention factor @ V (the values of the characters stacked together), is in turn a weighted mixture of all the values, showing the information gathered from paying the calculated attention to other characters.

---

# Multi-headed attention mechanism
* Multi-headed attention is when we have n heads in parallel that extract different relations between characters. One head may extract semantics, another may extract grammar, and so on. They create outputs which are then concatenated together again to be passed forward. Suppose we want 6 heads and have an embedding dimension of 36. The conventional rule of thumb is to have an output dimension of 6 for each head, and when concatenated together fits nicely to n_embedding.

---

# What comes next?
The extracted information is passed to a projection layer that fits it to the embedding dimension (if it's not already), and mixes the concatenated information into a n_embedding tensor. 