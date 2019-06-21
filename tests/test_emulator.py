from app.emulator import find_ideal


def test_find_ideal_all():
    p = [4, 5, 9, 1, 8, 13, 17]
    # diff
    # [5, 9, 1, 8, 13, 17]
    # -
    # [4, 5, 9, 1, 8, 13]
    # =
    # [1, 4, -8, 7, 5, 4]

    # ideal - sum of all positive differences
    result = find_ideal(p, just_once=False)
    assert result == 21


def test_find_ideal_once():
    p = [4, 5, 9, 1, 8, 13, 17]
    #             ^-------------min
    #                       ^---max
    #                           (max-min)
    result = find_ideal(p, just_once=True)
    assert result == 16

    p = [17, 3, 12, 15, 23, 8, 1]
    #        ^--------------------min
    #                   ^---------max
    result = find_ideal(p, just_once=True)
    assert result == 20

    p = [22, 21, 20, 29, 21, 24, 3]
    #            ^------------------min
    #                ^--------------max
    result = find_ideal(p, just_once=True)
    assert result == 9
