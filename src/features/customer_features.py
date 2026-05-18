"""Features cadastrais."""
import numpy as np
import pandas as pd


def build_customer_features(cad: pd.DataFrame, ref_dates: pd.DataFrame) -> pd.DataFrame:
    """Junta cadastral com datas de referencia (id_cliente -> ref_date) e calcula derivadas.

    ref_dates: DataFrame com colunas id_cliente, ref_date.
    Retorna ref_dates + features cadastrais.
    """
    df = ref_dates.merge(cad, on="id_cliente", how="left")

    # idade
    df["idade"] = ((df["ref_date"] - df["data_nascimento"]).dt.days / 365.25).round(2)
    df["faixa_idade"] = pd.cut(
        df["idade"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18_25", "26_35", "36_45", "46_55", "56_65", "65_mais"],
    ).astype(str)

    # renda
    df["renda_mensal"] = df["renda_anual"] / 12.0
    df["dependentes"] = (df["qtd_membros_familia"] - 1).clip(lower=0)
    df["dependentes_por_renda_mensal"] = df["dependentes"] / df["renda_mensal"].replace(0, np.nan)
    df["filhos_por_membro"] = df["qtd_filhos"] / df["qtd_membros_familia"].replace(0, np.nan)

    # bens
    df["flag_possui_bens"] = (
        (df["possui_carro"].fillna("N") == "Y") | (df["possui_imovel"].fillna("N") == "Y")
    ).astype(int)

    # regiao
    df["nota_regiao_diff"] = df["nota_regiao_cliente"] - df["nota_regiao_cliente_cidade"]

    # missing flag para ocupacao (31% null no dataset)
    df["flag_ocupacao_missing"] = df["ocupacao"].isna().astype(int)
    df["ocupacao"] = df["ocupacao"].fillna("OUTROS")

    # idade fora do esperado
    df["flag_idade_fora_padrao"] = ((df["idade"] < 18) | (df["idade"] > 75)).astype(int)

    return df.drop(columns=["data_nascimento"])
