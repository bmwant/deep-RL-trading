from app.visualizer import show_step


def test_show_step_with_header():
    header = ['h1', 'h2', 'h3']
    data = [
        [3, 4, 5],
    ]
    show_step(data, header)

    data = [
        [3, 4, 5],
        [9, 8, 7],
        [18, 16, 20],
    ]
    show_step(data, header)


def test_show_without_header():
    pass
