from pytorch_lightning.callbacks import TQDMProgressBar


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_postfix_str(s='Da da da')
        return bar

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.set_description('sanity...')
        return bar
