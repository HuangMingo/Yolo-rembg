from dataclasses import replace
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.config import Settings
from app.services.background_remover import BackgroundRemover


class BackgroundRemoverTests(unittest.TestCase):
    def make_remover(self, settings, result):
        module = ModuleType("rembg")
        session = object()
        module.new_session = Mock(return_value=session)
        calls = Mock(return_value=result)

        def remove(image, decontaminate=False, **kwargs):
            return calls(image, decontaminate=decontaminate, **kwargs)

        module.remove = remove
        with patch.dict("sys.modules", {"rembg": module}):
            remover = BackgroundRemover(settings)
        return remover, module, calls, session

    def test_pil_api_reuses_session_and_preserves_soft_alpha(self):
        output = Image.new("RGBA", (10, 8), (120, 80, 30, 128))
        settings = replace(Settings(), post_process_mask=True)
        remover, module, calls, session = self.make_remover(settings, output)
        source = Image.new("RGB", output.size)
        for _ in range(2):
            result = remover.remove(source)
            self.assertEqual(result.getpixel((0, 0))[3], 128)
            self.assertEqual(result.size, source.size)
        module.new_session.assert_called_once_with(settings.rembg_model)
        self.assertEqual(calls.call_count, 2)
        args, kwargs = calls.call_args
        self.assertIsInstance(args[0], Image.Image)
        self.assertIs(kwargs["session"], session)
        self.assertTrue(kwargs["alpha_matting"])
        self.assertTrue(kwargs["post_process_mask"])

    def test_decontaminate_without_alpha_matting(self):
        settings = replace(Settings(), alpha_matting=False, decontaminate=True)
        remover, _, calls, _ = self.make_remover(
            settings, Image.new("RGBA", (10, 8), (120, 80, 30, 128))
        )
        remover.remove(Image.new("RGB", (10, 8)))
        self.assertTrue(calls.call_args.kwargs["decontaminate"])

    def test_unsupported_decontaminate_has_explicit_error(self):
        module = ModuleType("rembg")
        module.remove = lambda image, **kwargs: image
        module.new_session = Mock()
        settings = replace(Settings(), alpha_matting=False, decontaminate=True)
        with patch.dict("sys.modules", {"rembg": module}):
            with self.assertRaisesRegex(RuntimeError, "chưa hỗ trợ"):
                BackgroundRemover(settings)
        module.new_session.assert_not_called()

    def test_alpha_matting_does_not_require_new_decontaminate_api(self):
        module = ModuleType("rembg")
        calls = Mock(return_value=Image.new("RGBA", (10, 8), (120, 80, 30, 128)))

        def remove(image, **kwargs):
            self.assertNotIn("decontaminate", kwargs)
            return calls(image, **kwargs)

        module.remove = remove
        module.new_session = Mock(return_value=object())
        settings = replace(Settings(), alpha_matting=True, decontaminate=True)
        with patch.dict("sys.modules", {"rembg": module}):
            remover = BackgroundRemover(settings)
        remover.remove(Image.new("RGB", (10, 8)))
        calls.assert_called_once()

    def test_unexpected_output_is_rejected(self):
        for output in (b"invalid result", Image.new("RGBA", (5, 4))):
            with self.subTest(output=type(output).__name__):
                remover, _, _, _ = self.make_remover(Settings(), output)
                with self.assertRaises(RuntimeError):
                    remover.remove(Image.new("RGB", (10, 8)))

    def test_empty_mask_and_inference_errors_are_reported(self):
        remover, _, calls, _ = self.make_remover(
            Settings(), Image.new("RGBA", (10, 8), (0, 0, 0, 0))
        )
        with self.assertRaisesRegex(RuntimeError, "mask"):
            remover.remove(Image.new("RGB", (10, 8)))
        calls.side_effect = ValueError("bad inference")
        with self.assertRaisesRegex(RuntimeError, "bad inference"):
            remover.remove(Image.new("RGB", (10, 8)))

    def test_invalid_matting_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(Settings(), alpha_foreground_threshold=270)
        with self.assertRaises(ValueError):
            replace(Settings(), alpha_background_threshold=240)


if __name__ == "__main__":
    unittest.main()
