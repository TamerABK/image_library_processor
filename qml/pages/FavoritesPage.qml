pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "favoritesPage"
    property var favorites: null
    property var inspector: null
    property var exporter: null
    property string viewSize: "Medium"
    Room36Theme.Theme { id: theme }
    Component.onDestruction: if (inspector) inspector.close()

    ColumnLayout {
        anchors.fill: parent
        spacing: 8
        RowLayout {
            Layout.fillWidth: true
            Item { Layout.fillWidth: true }
            Components.PillButton {
                objectName: "favoritesExportSelected"
                variant: "accent"
                text: "Export selected"
                enabled: root.exporter !== null && !root.exporter.busy && !root.exporter.preparing
                    && root.favorites !== null && root.favorites.photoModel.selectedCount > 0
                onClicked: root.exporter.prepareSelected(root.favorites.photoModel)
                Accessible.name: "Export selected favorite photos"
            }
        }
        Text {
            objectName: "favoritesMessage"
            Layout.fillWidth: true
            visible: root.favorites !== null && root.favorites.error.length > 0
            text: root.favorites ? root.favorites.error : ""
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            Accessible.name: text
        }
        Components.PhotoGrid {
            objectName: "favoritesGrid"
            Layout.fillWidth: true
            Layout.fillHeight: true
            photoModel: root.favorites ? root.favorites.photoModel : null
            favoritesController: root.favorites
            viewSize: root.viewSize
            openOnClick: true
            emptyText: root.favorites && root.favorites.loaded
                ? "No favorite photos yet. Use the heart on a photo to add it here."
                : root.favorites && root.favorites.error.length > 0
                    ? "Favorites are unavailable." : "Loading favorites…"
            onPhotoActivated: function(identity) {
                if (root.inspector && root.favorites)
                    root.inspector.openFromModel(root.favorites.photoModel, identity)
            }
        }
    }
    Component {
        id: favoriteInspectorDetails
        ColumnLayout {
            Text { text: "Favorite photo"; color: theme.mainBlue; font.pixelSize: 20 }
            Text { text: "Favorites are saved for this project."; color: theme.mutedText }
        }
    }
    Components.PhotoInspector {
        anchors.fill: parent
        controller: root.inspector
        favoritesController: root.favorites
        detailsComponent: favoriteInspectorDetails
        showReviewActions: false
        visible: controller !== null && controller.opened
    }
}
