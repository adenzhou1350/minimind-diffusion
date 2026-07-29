from trainer.trainer_utils import get_lr


def test_get_lr_cosine():
    # step=0: peak * 0.1 + 0.45 * 2 = 0.1 + 0.9 = 1.0 -> lr
    assert abs(get_lr(0, 100, 1e-3) - 1e-3) < 1e-9
    # step=total/2: cos(pi/2)=0 -> 0.1 + 0.45 = 0.55 * lr
    assert abs(get_lr(50, 100, 1e-3) - 0.55e-3) < 1e-9
    # step=total: cos(pi)=-1 -> 0.1 + 0 = 0.1 * lr
    assert abs(get_lr(100, 100, 1e-3) - 0.1e-3) < 1e-9


def test_get_lr_floor():
    # 不会低于 0.1 * lr
    assert get_lr(999, 1000, 1e-3) >= 0.1e-3
