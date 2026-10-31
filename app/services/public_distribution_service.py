"""Compare values with a descriptive public-data band, never a clinical range."""

from app.core.public_distribution import NHANES_PUBLIC_BANDS
from app.schemas.public_distribution import (
    PublicDistributionComparisonItem,
    PublicDistributionComparisonRequest,
    PublicDistributionComparisonResponse,
)
from ml.preprocessing.biopro import FEATURE_METADATA


def compare_with_public_distribution(
    request: PublicDistributionComparisonRequest,
) -> PublicDistributionComparisonResponse:
    items = []
    for item in request.items:
        lower, upper = NHANES_PUBLIC_BANDS[item.feature_code]
        if item.result < lower:
            position = "below"
            statement = "O valor está abaixo do percentil 5 desta distribuição pública."
        elif item.result > upper:
            position = "above"
            statement = "O valor está acima do percentil 95 desta distribuição pública."
        else:
            position = "within"
            statement = "O valor está entre os percentis 5 e 95 desta distribuição pública."
        items.append(
            PublicDistributionComparisonItem(
                feature_code=item.feature_code,
                test_name=FEATURE_METADATA[item.feature_code]["name"],
                result=item.result,
                unit=item.unit,
                band_lower=lower,
                band_upper=upper,
                position=position,
                statement=statement,
            )
        )
    return PublicDistributionComparisonResponse(
        items=items,
        limitation=(
            "Esta é uma faixa estatística descritiva de dados públicos NHANES, sem "
            "ponderação amostral. Não é intervalo clínico de referência, não define "
            "normalidade e não deve orientar decisões de saúde."
        ),
    )
