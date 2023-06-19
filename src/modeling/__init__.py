from src.modeling import matching
from src.modeling import mcvae

def _get_matching(args):
    return matching.MatchingMixin.create(args.model_name, args.model_path, args.use_symmetric_loss)

def _get_mcvae(args):
    return mcvae.DistilBertMCVAE.from_pretrained(
        args.model_path, 
        use_symmetric_loss=args.use_symmetric_loss, 
        z=args.z,
        kld_weight=args.kld_weight,
        use_kld_annealling=args.use_kld_annealling,
        kld_annealling_steps=args.kld_annealling_steps,
        use_message_prior=args.use_message_prior
    )


def get_model(args):
    models = {
        "matching": _get_matching,
        "mcvae": _get_mcvae,
    }

    return models[args.model_type](args)


from transformers import BertModel, DistilBertModel

def get_model_for_agent(model_type, model_path, z=None):
    if model_type == "bert":
        model = BertModel.from_pretrained(model_path)
    elif model_type == "distilbert":
        model = DistilBertModel.from_pretrained(model_path)
    elif model_type == "cvae":
        model = mcvae.DistilBertMCVAE.from_pretrained(model_path, z=z)

    else:
        raise ValueError("Model type not recognised")
    return model
