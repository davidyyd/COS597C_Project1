# r44c + document-so-far mean embedding in the contextual x0
Parent: r44c-r42b-unet (adds r45c's mechanism). Type: exploit.
The stack's first two layers are local (conv, 128-window attention), so early layers have no topic
context; a causal mean of the document's embeddings so far, re-injected with x0, provides it cheaply.
Confirm: < 0.932038 within the cap.
