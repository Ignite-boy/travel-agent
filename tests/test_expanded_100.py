import pytest

from app.agent.intent import parse_intent_json
from app.agent.planner import parse_plan
from app.memory.long_term import parse_preference_json
from app.agent.tools.flight_hotel_tool import search_flights_hotels
from app.agent.tools.weather_tool import get_weather


@pytest.mark.parametrize(
    "raw,city,days,budget,clear",
    [
        (f'{{"city":"{city}","days":{days},"budget":"{budget}","clear":true,"clarification_question":null}}', city, days, budget, True)
        for city, days, budget in [
            ("Tokyo", 1, "low"), ("Tokyo", 2, "medium"), ("Tokyo", 7, "high"),
            ("Paris", 1, "low"), ("Paris", 3, "medium"), ("Paris", 10, "high"),
            ("Delhi", 2, "low"), ("Delhi", 5, "medium"), ("Delhi", 14, "high"),
            ("Dubai", 1, "low"), ("Dubai", 4, "medium"), ("Dubai", 8, "high"),
            ("Singapore", 2, "low"), ("Singapore", 6, "medium"), ("Singapore", 12, "high"),
        ]
        + [
            ('{"city":null,"days":2,"budget":"medium","clear":false,"clarification_question":"Which city?"}', None, 2, "medium", False),
            ('{"city":"Tokyo","days":0,"budget":null,"clear":true,"clarification_question":null}', "Tokyo", 2, "medium", True),
            ('{"city":"","days":5,"budget":"high","clear":true,"clarification_question":null}', "", 5, "high", False),
            ('{"city":"Tokyo","days":3,"budget":"","clear":true,"clarification_question":"ok"}', "Tokyo", 3, "medium", True),
            ('{"city":"Tokyo","days":null,"budget":"low","clear":true,"clarification_question":null}', "Tokyo", 2, "low", True),
            ('{"city":"Tokyo","days":4,"budget":"high","clear":false,"clarification_question":"Need dates"}', "Tokyo", 4, "high", False),
            ('{"city":"New York","days":3,"budget":"medium","clear":1,"clarification_question":null}', "New York", 3, "medium", True),
            ('{"city":"Mumbai","days":7,"budget":"low","clear":"yes","clarification_question":null}', "Mumbai", 7, "low", True),
            ('{"city":"Rome","days":2,"budget":"medium","clear":false,"clarification_question":"How many days?"}', "Rome", 2, "medium", False),
            ('{"city":"Seoul","days":9,"budget":"high","clear":true,"clarification_question":"ignored"}', "Seoul", 9, "high", True),
            ("not json", None, 2, "medium", False),
            ("", None, 2, "medium", False),
            ("null", None, 2, "medium", False),
            ("[]", None, 2, "medium", False),
            ("{bad", None, 2, "medium", False),
        ],
    ids=lambda x: str(x[1]) if isinstance(x, tuple) else None,
)
def test_parse_intent_expanded(raw, city, days, budget, clear):
    result = parse_intent_json(raw)
    assert result["city"] == city
    assert result["days"] == days
    assert result["budget"] == budget
    assert result["clear"] is clear


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1. Check weather", ["Check weather"]),
        ("1) Check weather", ["Check weather"]),
        ("- Check weather", ["Check weather"]),
        ("* Check weather", ["Check weather"]),
        ("1. Check\n2. Hotels", ["Check", "Hotels"]),
        ("1) Check\n2) Hotels", ["Check", "Hotels"]),
        ("- Check\n- Hotels", ["Check", "Hotels"]),
        ("* Check\n* Hotels", ["Check", "Hotels"]),
        ("  1.  Check weather  ", ["Check weather"]),
        ("Check weather\n\nFind hotels", ["Check weather", "Find hotels"]),
        (["Check weather", "Find hotels"], ["Check weather", "Find hotels"]),
        (["1. Check weather", "2. Find hotels"], ["1. Check weather", "2. Find hotels"]),
        ("1. A\n2. B\n3. C\n4. D", ["A", "B", "C", "D"]),
        ("01. A\n02. B", ["A", "B"]),
        ("10) A\n11) B", ["A", "B"]),
        ("*** A", ["A"]),
        ("--- A", ["A"]),
        ("A", ["A"]),
        ("A\n B\n  C", ["A", "B", "C"]),
        ("1.\n2.", []),
        ("\n\n", []),
        (123, ["123"]),
        (None, ["None"]),
        ("1. Mumbai trip\n2) Delhi trip\n- Jaipur trip", ["Mumbai trip", "Delhi trip", "Jaipur trip"]),
        ("* museums\n* food\n* culture\n* parks", ["museums", "food", "culture", "parks"]),
    ],
)
def test_parse_plan_expanded(raw, expected):
    assert parse_plan(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('[]', []),
        ('["vegetarian"]', ["vegetarian"]),
        ('["museum", "nightlife"]', ["museum", "nightlife"]),
        ('[" budget traveller "]', ["budget traveller"]),
        ('["a", "", "b"]', ["a", "b"]),
        ('["a", "  ", "b"]', ["a", "b"]),
        ('[1, "a", true]', ["a"]),
        ('{"preference":"vegetarian"}', []),
        ('null', []),
        ('"vegetarian"', []),
        ('not json', []),
        ('', []),
        ('["a", "a", "b"]', ["a", "a", "b"]),
        ('["Mumbai food", "quiet hotels", "walking"]', ["Mumbai food", "quiet hotels", "walking"]),
        ('["Hindi", "vegetarian", "family-friendly"]', ["Hindi", "vegetarian", "family-friendly"]),
        ('```json\n["museum"]\n```', ["museum"]),
        ('```["museum"]```', ["museum"]),
        ('["one", null, "two"]', ["one", "two"]),
        ('[" x ", " y "]', ["x", "y"]),
        ('[true, false, 1, 2]', []),
    ],
)
def test_parse_preference_expanded(raw, expected):
    assert parse_preference_json(raw) == expected


@pytest.mark.parametrize("city", ["Tokyo", "Paris", "Delhi"])
def test_flight_hotel_known_city_contains_city(city):
    result = search_flights_hotels.invoke({"city": city, "budget": "medium"})
    assert city in result
    assert "mock data" in result


@pytest.mark.parametrize("budget", ["low", "medium", "high"])
def test_flight_hotel_budget_is_reflected(budget):
    result = search_flights_hotels.invoke({"city": "Tokyo", "budget": budget})
    assert budget in result.lower()


@pytest.mark.parametrize("city", ["Tokyo", "Paris", "Singapore"])
def test_weather_without_key_returns_city_fallback(monkeypatch, city):
    monkeypatch.setattr("app.agent.tools.weather_tool.settings.OPENWEATHER_API_KEY", "")
    result = get_weather.invoke({"city": city})
    assert city in result
    assert "mock weather" in result


@pytest.mark.parametrize("budget_pair", [("low", "high"), ("low", "medium"), ("medium", "high")])
def test_flight_hotel_budget_pairs_differ(budget_pair):
    first, second = budget_pair
    r1 = search_flights_hotels.invoke({"city": "Paris", "budget": first})
    r2 = search_flights_hotels.invoke({"city": "Paris", "budget": second})
    assert r1 != r2
